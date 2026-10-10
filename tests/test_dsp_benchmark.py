import copy
import json
import wave

import numpy as np
import pytest

from scripts import benchmark_dsp_gtsinger as benchmark


def write_pcm(path, values, width=2, rate=48000):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=np.int64)
    if values.ndim == 1:
        values = values[:, None]
    if width == 3:
        flat = values.reshape(-1)
        raw = np.column_stack((flat & 255, (flat >> 8) & 255, (flat >> 16) & 255)).astype(np.uint8).tobytes()
    elif width in (2, 4):
        raw = values.astype('<i' + str(width)).tobytes()
    else:
        raw = values.astype(np.uint8).tobytes()
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(values.shape[1])
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(raw)


def audio_segment(path, rate=48000, frames=4):
    return {'audio_path': str(path), 'sample_rate': rate, 'num_samples': frames,
            'sources': {'wav': {'sha256': benchmark.digest(path)}}}


@pytest.mark.parametrize('width', [2, 3, 4])
@pytest.mark.parametrize('channels', [1, 2])
def test_integer_pcm_is_on_original_grid_and_uses_first_channel(tmp_path, width, channels):
    scale = 1 << (width * 8 - 1)
    first = np.array([-scale, -1, 0, scale - 1])
    samples = first if channels == 1 else np.column_stack((first, np.array([3, 2, 1, 0])))
    path = tmp_path / 'audio.wav'
    write_pcm(path, samples, width)
    decoded = benchmark.read_audio(audio_segment(path))
    assert decoded.dtype == np.float64 and decoded.shape == (4,)
    np.testing.assert_array_equal(decoded, first / scale)


def test_unsupported_pcm_is_rejected(tmp_path):
    path = tmp_path / 'eightbit.wav'
    write_pcm(path, [1, 2, 3, 4], width=1)
    with pytest.raises(ValueError, match='PCM16/24/32'):
        benchmark.read_audio(audio_segment(path))


@pytest.mark.parametrize('change', ['hash', 'rate', 'frames'])
def test_decoder_verifies_hash_and_original_sample_grid(tmp_path, change):
    path = tmp_path / 'audio.wav'
    write_pcm(path, [0, 1, 2, 3])
    segment = audio_segment(path)
    if change == 'hash':
        segment['sources']['wav']['sha256'] = '0' * 64
    elif change == 'rate':
        segment['sample_rate'] = 44100
    else:
        segment['num_samples'] = 5
    with pytest.raises(ValueError, match='frozen source|sample grid'):
        benchmark.read_audio(segment)


def summary(errors, total=None):
    return benchmark.summarize([{'total': len(errors) if total is None else total,
                                'missing': 0, 'text_mismatch': 0,
                                'errors': [{'absolute_error_ms': error} for error in errors]}])


def validation(baseline, trial_errors):
    return {'baseline': baseline, 'trials': {
        config.name: {'config': config.to_dict(), 'metrics': summary(errors), 'by_model': {}}
        for config, errors in zip(benchmark.CONFIG_GRID, trial_errors)}}


def test_choice_uses_one_pooled_config_and_declared_order_for_ties():
    result = benchmark.choose_configuration(validation(summary([40, 40]), [[20, 20], [20, 20], [30, 30]]))
    assert result['best_trial_config'] == benchmark.CONFIG_GRID[0].to_dict()
    assert result['selected'] == 'dsp'
    assert result['selected_config'] == result['best_trial_config']
    assert all(result['decision']['gates'].values())


@pytest.mark.parametrize('baseline,errors,gate', [
    ([20, 20], [20, 20], 'strict_mae_improvement'),
    ([0, 100], [20, 20], 'coverage_not_decreased'),
    ([490, 490], [0, 501], 'severe_not_increased'),
])
def test_failed_gate_retains_coarse_but_still_records_best_trial(baseline, errors, gate):
    result = benchmark.choose_configuration(validation(summary(baseline), [errors] * 3))
    assert result['selected'] == 'none' and result['selected_config'] is None
    assert result['best_trial_config'] == benchmark.CONFIG_GRID[0].to_dict()
    assert result['decision']['gates'][gate] is False


def test_losing_evaluated_boundaries_cannot_improve_selection_by_disappearing():
    evidence = validation(summary([100, 100]), [[0], [0], [0]])
    for trial in evidence['trials'].values():
        trial['metrics'] = summary([0], total=2)
    result = benchmark.choose_configuration(evidence)
    assert result['selected'] == 'none'
    assert result['decision']['gates']['evaluated_not_decreased'] is False


def test_no_evaluated_validation_reference_cannot_fabricate_selection():
    with pytest.raises(ValueError, match='No evaluated validation'):
        benchmark.choose_configuration(validation(summary([]), [[0], [0], [0]]))


@pytest.fixture
def dataset(tmp_path):
    audio_root = tmp_path / 'audio'
    segments, assignments, coarse_rows = [], [], []
    for index, split in enumerate(('train', 'val', 'test')):
        segment_id = 'English/singer-{}/song-{}/0000'.format(index, index)
        path = audio_root / (segment_id + '.wav')
        write_pcm(path, [index, 1, 2, 3])
        segment = dict(audio_segment(path), segment_id=segment_id, song_id='song-' + str(index),
                       singer='singer-' + str(index), language='English', units=[
                           {'kind': 'lyric', 'text': 'one', 'boundary': {
                               'source': 'textgrid:first_phone', 'candidate_sample': 1}}])
        segments.append(segment)
        assignments.append({'segment_id': segment_id, 'canonical_song_id': segment['song_id'], 'split': split})
        candidate = {'text': 'one', 'start_sample': 2, 'start_seconds': 2 / 48000,
                     'original_sample_rate': 48000, 'original_num_samples': 4,
                     'status': 'candidate', 'confidence': None}
        coarse_rows.append({'segment_id': segment_id, 'song_id': segment['song_id'], 'split': split,
                            'runs': {method + ':legacy': {'status': 'completed', 'candidates': [copy.deepcopy(candidate)]}
                                     for method in benchmark.METHODS}})
    segments_path, splits_path, coarse_path = [tmp_path / name for name in ('segments.jsonl', 'splits.json', 'coarse.json')]
    def save():
        segments_path.write_text(''.join(json.dumps(row) + '\n' for row in segments))
        splits_path.write_text(json.dumps({'identity_mode': 'dataset', 'held_out_singers': [], 'assignments': assignments}))
        coarse_path.write_text(json.dumps({'schema_version': 1, 'status': 'completed',
                                          'input_sha256': {str(segments_path): benchmark.digest(segments_path),
                                                           str(splits_path): benchmark.digest(splits_path)},
                                          'rows': coarse_rows}))
    save()
    return {'segments': segments, 'assignments': assignments, 'coarse_rows': coarse_rows,
            'segments_path': segments_path, 'splits_path': splits_path, 'coarse_path': coarse_path,
            'audio_root': audio_root, 'save': save, 'root': tmp_path}


def args(dataset, out_dir, selection=None):
    values = ['--segments', str(dataset['segments_path']), '--splits', str(dataset['splits_path']),
              '--audio-root', str(dataset['audio_root']), '--coarse-report', str(dataset['coarse_path']),
              '--out-dir', str(out_dir)]
    return values + (['--selection-file', str(selection)] if selection else [])


@pytest.fixture
def fake_dsp(monkeypatch):
    calls = []
    def features(audio, sample_rate, hop_samples):
        calls.append(('features', len(audio), sample_rate, hop_samples))
        return {'sample_positions': np.array([0]), 'features': {}, 'metadata': {'sample_rate': sample_rate}}
    def refine(audio, sample_rate, candidates, config, features):
        calls.append(('refine', config.name, copy.deepcopy(candidates)))
        result = copy.deepcopy(candidates)
        for row in result:
            row.update(start_sample=1, start_seconds=1 / sample_rate,
                       refinement={'status': 'refined', 'reason': None, 'delta_samples': -1})
        return result
    monkeypatch.setattr(benchmark, 'extract_features', features)
    monkeypatch.setattr(benchmark, 'refine_boundaries', refine)
    return calls


def read_selection(out_dir):
    return json.loads((out_dir / 'selection.json').read_text())


def test_select_does_not_read_train_or_test_source_labels(dataset, fake_dsp):
    dataset['segments'][0]['units'][0]['boundary'] = {'invalid': 'train label is not selection input'}
    dataset['segments'][2]['units'][0]['boundary'] = {'invalid': 'test label is not selection input'}
    dataset['save']()
    out = dataset['root'] / 'select'
    assert benchmark.cli(args(dataset, out)) == 0
    report = json.loads((out / 'result.json').read_text())
    assert [row['split'] for row in report['rows']] == ['val']
    assert len([call for call in fake_dsp if call[0] == 'features']) == 1
    assert len([call for call in fake_dsp if call[0] == 'refine']) == 6
    assert report['selection']['selected'] == 'dsp'
    assert all(key.endswith(':val') for key in report['summary'])


def test_changed_test_annotations_cannot_change_validation_choice(dataset, fake_dsp):
    first, second = dataset['root'] / 'first', dataset['root'] / 'second'
    benchmark.cli(args(dataset, first))
    dataset['segments'][2]['units'][0]['boundary']['candidate_sample'] = 3
    dataset['save']()
    benchmark.cli(args(dataset, second))
    a, b = read_selection(first), read_selection(second)
    assert a['validation'] == b['validation']
    assert a['best_trial_config'] == b['best_trial_config']
    assert a['selected'] == b['selected']
    assert a['input_sha256'] != b['input_sha256']


def test_selection_function_refuses_test_rows(dataset):
    segment = dict(dataset['segments'][2], split='test')
    with pytest.raises(ValueError, match='validation segments only'):
        benchmark.select_configuration([segment], {segment['segment_id']: dataset['coarse_rows'][2]})


@pytest.mark.parametrize('change', ['status', 'missing_row', 'duplicate_row', 'song', 'split', 'run', 'grid', 'digest'])
def test_coarse_report_contract_is_checked_before_output(dataset, change):
    report = json.loads(dataset['coarse_path'].read_text())
    if change == 'status':
        report['status'] = 'failed'
    elif change == 'missing_row':
        report['rows'].pop()
    elif change == 'duplicate_row':
        report['rows'].append(copy.deepcopy(report['rows'][0]))
    elif change in ('song', 'split'):
        report['rows'][0]['song_id' if change == 'song' else 'split'] = 'wrong'
    elif change == 'run':
        report['rows'][0]['runs']['mtl:legacy']['status'] = 'failed'
    elif change == 'grid':
        report['rows'][0]['runs']['mtl:legacy']['candidates'][0]['original_sample_rate'] = 44100
    else:
        report['input_sha256'][str(dataset['segments_path'])] = '0' * 64
    dataset['coarse_path'].write_text(json.dumps(report))
    out = dataset['root'] / 'invalid'
    with pytest.raises(SystemExit) as error:
        benchmark.cli(args(dataset, out))
    assert error.value.code == 1 and not out.exists()


@pytest.mark.parametrize('change', ['input', 'runtime', 'grid', 'policy', 'trial', 'decision'])
def test_frozen_selection_binding_and_decision_are_verified(dataset, fake_dsp, change):
    out = dataset['root'] / 'select'
    benchmark.cli(args(dataset, out))
    selection = read_selection(out)
    original = copy.deepcopy(selection)
    if change == 'input':
        selection['input_sha256']['segments'] = '0' * 64
    elif change == 'runtime':
        selection['runtime_sha256']['t2l/boundary/dsp.py'] = '0' * 64
    elif change == 'grid':
        selection['config_grid'][0]['prior_weight'] = .3
    elif change == 'policy':
        selection['policy']['selection_split'] = 'test'
    elif change == 'trial':
        selection['best_trial_config'] = benchmark.CONFIG_GRID[1].to_dict()
    else:
        selection['selected'] = 'none'
    with pytest.raises(ValueError, match='Selection'):
        benchmark.verify_selection(selection, original['input_sha256'], original['runtime_sha256'])


def test_evaluate_runs_only_fixed_trial_keeps_selection_file_and_does_not_reselect(dataset, fake_dsp, monkeypatch):
    selected, evaluated = dataset['root'] / 'select', dataset['root'] / 'evaluate'
    benchmark.cli(args(dataset, selected))
    selection_path = selected / 'selection.json'
    original = selection_path.read_bytes()
    fake_dsp.clear()
    def never_select(*unused):
        raise AssertionError('Evaluation must not choose a configuration')
    monkeypatch.setattr(benchmark, 'select_configuration', never_select)
    assert benchmark.cli(args(dataset, evaluated, selection_path)) == 0
    assert selection_path.read_bytes() == original
    assert not (evaluated / 'selection.json').exists()
    refiner_calls = [call for call in fake_dsp if call[0] == 'refine']
    assert len(refiner_calls) == 6
    assert {call[1] for call in refiner_calls} == {'conservative'}
    report = json.loads((evaluated / 'result.json').read_text())
    assert report['stage'] == 'evaluate' and {row['split'] for row in report['rows']} == {'train', 'val', 'test'}
    assert set(report['summary']) == {method + ':' + variant + ':' + split for method in benchmark.METHODS
                                     for variant in ('coarse', 'dsp_trial', 'selected') for split in ('train', 'val', 'test')}
    assert all(report['summary'][method + ':selected:test'] == report['summary'][method + ':dsp_trial:test']
               for method in benchmark.METHODS)
    assert report['gpu'] is None and report['independent_human_accuracy'] is False


def test_production_none_still_evaluates_trial_and_selected_reuses_coarse(dataset, fake_dsp, monkeypatch):
    def unchanged(audio, sample_rate, candidates, config, features):
        return [dict(copy.deepcopy(row), refinement={'status': 'fallback', 'reason': 'weak_acoustic_evidence',
                                                     'delta_samples': 0}) for row in candidates]
    monkeypatch.setattr(benchmark, 'refine_boundaries', unchanged)
    selected, evaluated = dataset['root'] / 'select', dataset['root'] / 'evaluate'
    benchmark.cli(args(dataset, selected))
    assert read_selection(selected)['selected'] == 'none'
    benchmark.cli(args(dataset, evaluated, selected / 'selection.json'))
    report = json.loads((evaluated / 'result.json').read_text())
    for row in report['rows']:
        for runs in row['runs'].values():
            assert runs['selected']['variant'] == 'coarse'
            assert runs['dsp_trial']['fallback_reasons'] == {'weak_acoustic_evidence': 1}
            assert runs['dsp_trial']['candidates'][0]['refinement']['status'] == 'fallback'
    assert all(report['summary'][method + ':selected:test'] == report['summary'][method + ':coarse:test']
               for method in benchmark.METHODS)


def test_mismatched_and_missing_candidates_remain_in_all_boundary_denominator(dataset, fake_dsp):
    row = dataset['coarse_rows'][2]
    row['runs']['mtl:legacy']['candidates'][0].update(start_sample=None, start_seconds=None, status='unaligned')
    row['runs']['mtl_bdr:legacy']['candidates'][0]['text'] = 'wrong'
    dataset['save']()
    selected, evaluated = dataset['root'] / 'select', dataset['root'] / 'evaluate'
    benchmark.cli(args(dataset, selected))
    benchmark.cli(args(dataset, evaluated, selected / 'selection.json'))
    report = json.loads((evaluated / 'result.json').read_text())
    a, b = [report['summary'][method + ':coarse:test'] for method in benchmark.METHODS]
    assert a['total'] == b['total'] == 1 and a['evaluated'] == b['evaluated'] == 0
    assert a['missing'] == b['text_mismatch'] == 1
    assert a['coverage_of_all']['5'] == b['coverage_of_all']['5'] == 0
    assert report['failure_count'] >= 2 and len(report['failures']) <= 50


def test_evaluate_rejects_changed_test_input_before_output(dataset, fake_dsp):
    selected, evaluated = dataset['root'] / 'select', dataset['root'] / 'evaluate'
    benchmark.cli(args(dataset, selected))
    dataset['segments'][2]['units'][0]['boundary']['candidate_sample'] = 3
    dataset['save']()
    with pytest.raises(SystemExit) as error:
        benchmark.cli(args(dataset, evaluated, selected / 'selection.json'))
    assert error.value.code == 1 and not evaluated.exists()


def test_explicit_null_selection_cannot_silently_run_selection(dataset):
    selection = dataset['root'] / 'null-selection.json'
    selection.write_text('null')
    out = dataset['root'] / 'invalid-selection'
    with pytest.raises(SystemExit) as error:
        benchmark.cli(args(dataset, out, selection))
    assert error.value.code == 1 and not out.exists()


def test_runtime_change_during_run_does_not_publish_completed_selection(dataset, fake_dsp, monkeypatch):
    original = benchmark.runtime_digests()
    calls = []
    def runtime():
        calls.append(True)
        return original if len(calls) == 1 else dict(original, changed='0' * 64)
    monkeypatch.setattr(benchmark, 'runtime_digests', runtime)
    out = dataset['root'] / 'runtime-changed'
    with pytest.raises(SystemExit) as error:
        benchmark.cli(args(dataset, out))
    assert error.value.code == 1 and len(calls) == 2
    assert not (out / 'selection.json').exists()
    report = json.loads((out / 'result.json').read_text())
    assert report['status'] == 'failed' and 'changed during benchmark' in report['error']['message']


def test_output_is_exclusive_and_failure_is_persisted(dataset, fake_dsp, monkeypatch):
    out = dataset['root'] / 'select'
    out.mkdir()
    marker = out / 'keep.txt'
    marker.write_text('existing')
    with pytest.raises(SystemExit):
        benchmark.cli(args(dataset, out))
    assert marker.read_text() == 'existing' and not (out / 'result.json').exists()
    failed = dataset['root'] / 'failed'
    def failure(*args, **kwargs):
        raise ValueError('Synthetic feature failure')
    monkeypatch.setattr(benchmark, 'extract_features', failure)
    with pytest.raises(SystemExit):
        benchmark.cli(args(dataset, failed))
    report = json.loads((failed / 'result.json').read_text())
    assert report['status'] == 'failed' and report['error']['message'] == 'Synthetic feature failure'
    assert report['config_grid'] == [config.to_dict() for config in benchmark.CONFIG_GRID]
    assert not (failed / 'selection.json').exists()
    assert not list(failed.glob('*.tmp'))
