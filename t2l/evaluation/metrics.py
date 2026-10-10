"""Grouped summaries and song-cluster bootstrap on frozen predicted boundaries."""
from collections import defaultdict

import numpy as np

from scripts.benchmark_coarse_gtsinger import source_errors, summarize


def classified_records(segment, candidates):
    metrics = source_errors(segment, candidates)
    errors = {e['unit_index']: e for e in metrics['errors']}
    records = []
    for index, unit in enumerate(segment['units']):
        if unit['kind'] != 'lyric':
            continue
        classes = unit.get('classifications', {})
        record = {key: segment.get(key, 'unknown') for key in
                  ('segment_id', 'song_id', 'split', 'language', 'singer', 'technique', 'group')}
        record.update(unit_index=index, text=unit['text'],
                      first_phoneme=classes.get('first_phoneme', {}).get('value', 'unknown'))
        for name in ('breathy', 'glissando', 'long_note', 'connected_singing', 'resinging'):
            record[name] = classes.get(name, {}).get('value', 'unknown')
        if index in errors:
            record.update(status='evaluated', **errors[index])
        elif metrics['text_mismatch'] and len(candidates) != metrics['total']:
            record['status'] = 'text_mismatch'
        else:
            lyric_position = sum(u['kind'] == 'lyric' for u in segment['units'][:index])
            from scripts.benchmark_coarse_gtsinger import text_key
            candidate = candidates[lyric_position]
            record['status'] = ('text_mismatch' if not text_key(unit['text'])
                                or text_key(candidate['text']) != text_key(unit['text'])
                                else 'missing')
        records.append(record)
    return records


def record_summary(records):
    source = {'total': len(records), 'missing': sum(r['status'] == 'missing' for r in records),
              'text_mismatch': sum(r['status'] == 'text_mismatch' for r in records),
              'errors': [r for r in records if r['status'] == 'evaluated']}
    result = summarize([source])
    total = result['total']
    result['missing_rate'] = result['missing'] / total if total else None
    result['text_mismatch_rate'] = result['text_mismatch'] / total if total else None
    result['song_count'] = len({r['song_id'] for r in records})
    return result


def grouped_summary(records, fields=('language', 'first_phoneme', 'technique', 'breathy',
                                      'glissando', 'long_note', 'singer', 'group')):
    result = {}
    for field in fields:
        groups = defaultdict(list)
        for record in records:
            groups[str(record.get(field, 'unknown'))].append(record)
        result[field] = {key: record_summary(values) for key, values in sorted(groups.items())}
    return result


def song_bootstrap(records, *, replicates=2000, seed=7):
    """Resample entire songs, including every failure; never individual boundaries."""
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise ValueError('replicates must be an integer >=100')
    songs = defaultdict(list)
    for record in records:
        songs[record['song_id']].append(record)
    keys = sorted(songs)
    result = {'unit': 'song', 'replicates': replicates, 'seed': seed, 'song_count': len(keys),
              'method': 'percentile 95% CI; pooled boundaries of resampled complete song clusters',
              'reference': 'published_dataset_annotation', 'independent_human_accuracy': False,
              'warning': 'Few songs give unstable intervals; this does not cover source annotation bias.',
              'point': record_summary(records), 'intervals': {}}
    if len(keys) < 2:
        result['status'] = 'insufficient_independent_songs'
        return result
    rng = np.random.default_rng(seed)
    statistics = defaultdict(list)
    for _ in range(replicates):
        draw = rng.integers(0, len(keys), size=len(keys))
        metrics = record_summary([record for index in draw for record in songs[keys[index]]])
        for name in ('mae_ms', 'median_ae_ms', 'p90_ms', 'p95_ms', 'missing_rate', 'text_mismatch_rate'):
            if metrics[name] is not None:
                statistics[name].append(metrics[name])
        for ms, value in metrics['coverage_of_all'].items():
            if value is not None:
                statistics['coverage_' + ms + 'ms'].append(value)
    for name, values in statistics.items():
        result['intervals'][name] = [float(x) for x in np.quantile(values, [.025, .975])]
    result['status'] = 'completed'
    return result
