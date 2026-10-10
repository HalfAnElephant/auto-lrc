import copy
import json

import pytest

from scripts import plot_boundary_results as plots


def reports():
    inputs = {'segments': '1' * 64, 'splits': '2' * 64, 'coarse_report': '3' * 64, 'config': '4' * 64}
    validation = {'mae_ms': 12., 'total': 3, 'evaluated': 3}
    training = {'schema_version': 1, 'protocol': 'boundary_training', 'status': 'completed',
        'reference': plots.REFERENCE, 'independent_human_accuracy': False, 'selection_split': 'val',
        'input_sha256': inputs, 'runtime_sha256': {'model.py': '5' * 64},
        'training': {'seed': 7, 'epochs': 60}, 'config': {'hop_seconds': .0025},
        'checkpoint': {'sha256': '6' * 64}, 'best_epoch': 2, 'best_validation': validation,
        'history': [{'epoch': 1, 'loss': .8, 'loss_parts': {'heatmap': .5, 'delta': .3},
                     'validation': {'mae_ms': 20.}},
                    {'epoch': 2, 'loss': .5, 'loss_parts': {'heatmap': .3, 'delta': .2},
                     'validation': copy.deepcopy(validation)}]}
    metadata = {name: copy.deepcopy(training[name]) for name in
                ('input_sha256', 'runtime_sha256', 'training', 'best_epoch')}
    metadata.update(validation=copy.deepcopy(validation), reference=plots.REFERENCE,
                    independent_human_accuracy=False, selection_split='val', training_splits=['train', 'val'])
    evaluation = {'schema_version': 1, 'protocol': 'boundary_evaluation', 'status': 'completed',
        'reference': plots.REFERENCE, 'independent_human_accuracy': False, 'selection_split': 'val',
        'test_used_for_selection': False, 'config': copy.deepcopy(training['config']),
        'input_sha256': {key: value for key, value in inputs.items() if key != 'config'},
        'checkpoint_sha256': training['checkpoint']['sha256'], 'training_metadata': metadata, 'rows': []}
    for split in ('train', 'val', 'test'):
        row = {'segment_id': split, 'song_id': split + '-song', 'split': split, 'runs': {}}
        for method in plots.METHODS:
            row['runs'][method] = {}
            for number, variant in enumerate(plots.BASE_VARIANTS):
                errors = [{'unit_index': 0, 'text': 'word', 'reference_sample': 24000,
                           'reference_seconds': .5, 'signed_error_ms': 100. - number * 10,
                           'absolute_error_ms': 100. - number * 10},
                          {'unit_index': 2, 'text': 'other', 'reference_sample': 48000,
                           'reference_seconds': 1., 'signed_error_ms': -20. - number,
                           'absolute_error_ms': 20. + number}]
                row['runs'][method][variant] = {'status': 'completed',
                    'source_metrics': {'total': 3, 'missing': 1, 'text_mismatch': 0, 'errors': errors}}
        evaluation['rows'].append(row)
    return training, evaluation


def test_prepare_retains_missing_denominator_and_unique_case_ranking():
    training, evaluation = reports()
    prepared = plots.prepare_results(training, evaluation)
    distribution = prepared['distributions']['mtl']['coarse']
    assert distribution['errors_ms'] == [20., 100.]
    assert distribution['cdf_fraction_of_all'] == [1 / 3, 2 / 3]
    assert distribution['missing'] == 1
    assert [(case['segment_id'], case['unit_index']) for case in prepared['failure_cases']] == [('test', 0), ('test', 2)]
    assert prepared['failure_cases'][0]['worst_variant'] == 'coarse'
    assert prepared['trajectory'][1]['val_mae_ms'] == 12.
    assert prepared['best_epoch'] == 2


def test_non_test_error_values_cannot_influence_distribution_or_case_selection():
    training, evaluation = reports()
    expected = plots.prepare_results(training, evaluation)
    for row in evaluation['rows']:
        if row['split'] != 'test':
            row['runs'] = {'unused': {'source_metrics': {'errors': 'not plot input'}}}
    assert plots.prepare_results(training, evaluation) == expected


def test_optional_fifth_variant_is_plotted_without_selection():
    training, evaluation = reports()
    for method in plots.METHODS:
        runs = evaluation['rows'][-1]['runs'][method]
        runs['dp'] = copy.deepcopy(runs['neural'])
    prepared = plots.prepare_results(training, evaluation)
    assert prepared['variants'] == list(plots.BASE_VARIANTS) + ['dp']
    assert prepared['distributions']['mtl']['dp']['total'] == 3


def test_failed_predictions_and_text_mismatch_remain_in_denominator():
    training, evaluation = reports()
    run = evaluation['rows'][-1]['runs']['mtl']['neural']
    run.update(status='failed', source_metrics={'total': 3, 'missing': 2, 'text_mismatch': 1, 'errors': []})
    prepared = plots.prepare_results(training, evaluation)
    assert prepared['distributions']['mtl']['neural'] == {
        'total': 3, 'missing': 2, 'text_mismatch': 1, 'errors_ms': [], 'cdf_fraction_of_all': []}


@pytest.mark.parametrize('change', [
    lambda t, e: t.update(status='running'),
    lambda t, e: e.update(status='failed'),
    lambda t, e: e.update(independent_human_accuracy=True),
    lambda t, e: e.update(test_used_for_selection=True),
    lambda t, e: e['input_sha256'].update(segments='7' * 64),
    lambda t, e: e.update(checkpoint_sha256='7' * 64),
    lambda t, e: e['training_metadata'].update(best_epoch=1),
    lambda t, e: e['training_metadata'].update(training_splits=['train', 'val', 'test']),
    lambda t, e: e['training_metadata']['runtime_sha256'].update(**{'model.py': '7' * 64}),
    lambda t, e: t['history'][0].update(loss=float('nan')),
    lambda t, e: t['history'][1].update(epoch=1),
    lambda t, e: t['history'][1]['validation'].update(mae_ms=10.),
    lambda t, e: e['rows'][-1]['runs']['mtl'].pop('dsp'),
    lambda t, e: e['rows'][-1]['runs']['mtl']['coarse']['source_metrics'].update(total=4),
    lambda t, e: e['rows'][-1]['runs']['mtl']['coarse']['source_metrics']['errors'][0].update(reference_sample=1),
])
def test_prepare_rejects_unbound_or_inconsistent_evidence(change):
    training, evaluation = reports()
    change(training, evaluation)
    with pytest.raises(ValueError):
        plots.prepare_results(training, evaluation)


def test_case_selection_is_six_unique_boundaries_with_identity_tie_break():
    training, evaluation = reports()
    original = evaluation['rows'][-1]
    for index in reversed(range(5)):
        row = copy.deepcopy(original)
        row['segment_id'] = 'test-' + str(index)
        evaluation['rows'].append(row)
    cases = plots.prepare_results(training, evaluation)['failure_cases']
    assert len(cases) == 6
    assert [case['segment_id'] for case in cases] == ['test', 'test-0', 'test-1', 'test-2', 'test-3', 'test-4']
    assert all(case['unit_index'] == 0 for case in cases)


def test_case_evidence_keeps_original_grid_and_neural_outputs():
    training, evaluation = reports()
    case = plots.prepare_results(training, evaluation)['failure_cases'][0]
    row = evaluation['rows'][-1]
    segment = {'segment_id': 'test', 'song_id': 'test-song', 'split': 'test', 'sample_rate': 48000,
        'num_samples': 96000, 'audio_path': '/original/test.wav', 'sources': {'wav': {'sha256': '8' * 64}},
        'units': [{'kind': 'lyric', 'text': 'word', 'boundary': {'source': 'textgrid:first_phone', 'candidate_sample': 24000}},
                  {'kind': 'silence', 'text': 'SP'},
                  {'kind': 'lyric', 'text': 'other', 'boundary': {'source': 'textgrid:first_phone', 'candidate_sample': 48000}},
                  {'kind': 'lyric', 'text': 'missing'}]}
    candidate = {'text': 'word', 'start_sample': 28800, 'start_seconds': .6, 'status': 'candidate',
        'original_sample_rate': 48000, 'original_num_samples': 96000,
        'refinement': {'status': 'refined', 'confidence_logit': -.3, 'top_candidates': [
            {'sample': 28800, 'seconds': .6, 'heatmap_mass': .4, 'heatmap_logit': 1.2}]}}
    for runs in row['runs'].values():
        for run in runs.values():
            run['candidates'] = [copy.deepcopy(candidate), {'text': 'other'}, {'text': 'missing'}]
    evidence = plots.case_evidence(case, row, segment, plots.BASE_VARIANTS)
    assert evidence['methods']['mtl']['neural'] == candidate
    assert evidence['source_annotation_sample'] == 24000
    assert evidence['wav_sha256'] == '8' * 64
    wrong = copy.deepcopy(row)
    wrong['runs']['mtl']['coarse']['candidates'][0]['start_seconds'] = .7
    with pytest.raises(ValueError, match='sample/time'):
        plots.case_evidence(case, wrong, segment, plots.BASE_VARIANTS)
    segment['units'][0]['boundary']['candidate_sample'] += 1
    with pytest.raises(ValueError, match='annotation differs'):
        plots.case_evidence(case, row, segment, plots.BASE_VARIANTS)


def test_display_text_avoids_missing_glyphs_without_altering_index_text():
    assert plots.display_text('word 中文') == 'word [U+4E2D][U+6587]'
    assert plots.display_text('a\nb') == 'a[U+000A]b'


@pytest.fixture
def cli_inputs(tmp_path, monkeypatch):
    training, evaluation = reports()
    paths = {name: tmp_path / (name + '.json') for name in ('training', 'evaluation')}
    paths['training'].write_text(json.dumps(training))
    paths['evaluation'].write_text(json.dumps(evaluation))
    def fake_render(prepared, out):
        (out / 'training-curves.png').write_bytes(b'frozen fixture plot')
        (out / 'test-error-cdf.png').write_bytes(b'frozen fixture CDF')
        return {'matplotlib': 'fixture'}
    monkeypatch.setattr(plots, 'render_overview', fake_render)
    return paths, training, evaluation


def cli_args(paths, out):
    return ['--training-report', str(paths['training']), '--evaluation-report', str(paths['evaluation']), '--out-dir', str(out)]


def test_overview_cli_needs_no_audio_and_binds_results_script_images(cli_inputs, tmp_path):
    paths, _, _ = cli_inputs
    out = tmp_path / 'plots'
    assert plots.cli(cli_args(paths, out)) == 0
    index = json.loads((out / 'index.json').read_text())
    assert index['status'] == 'completed'
    assert index['independent_human_accuracy'] is False
    assert index['input_sha256']['training_report'] == plots.digest(paths['training'])
    assert index['input_sha256']['script'] == plots.digest(plots.__file__)
    assert index['evaluation_input_sha256']['segments'] == '1' * 64
    assert index['images']['test-error-cdf.png'] == plots.digest(out / 'test-error-cdf.png')
    assert index['failure_cases'] == []
    assert index['distributions']['mtl']['coarse']['total'] == 3
    assert index['distributions']['mtl']['coarse']['evaluated'] == 2


def test_cli_never_clobbers_existing_output(cli_inputs, tmp_path):
    paths, _, _ = cli_inputs
    out = tmp_path / 'plots'
    out.mkdir()
    sentinel = out / 'index.json'
    sentinel.write_text('existing result')
    assert plots.cli(cli_args(paths, out)) == 1
    assert sentinel.read_text() == 'existing result'
    assert list(out.iterdir()) == [sentinel]


def test_cli_render_failure_saves_failed_index(cli_inputs, tmp_path, monkeypatch):
    paths, _, _ = cli_inputs
    out = tmp_path / 'plots'
    def fail(prepared, output):
        (output / 'training-curves.png').write_bytes(b'partial rendering')
        raise RuntimeError('fixture rendering failure')
    monkeypatch.setattr(plots, 'render_overview', fail)
    assert plots.cli(cli_args(paths, out)) == 1
    index = json.loads((out / 'index.json').read_text())
    assert index['status'] == 'failed'
    assert index['error']['message'] == 'fixture rendering failure'
    assert 'training-curves.png' in index['images']


def test_cli_rejects_input_change_during_render(cli_inputs, tmp_path, monkeypatch):
    paths, _, _ = cli_inputs
    out = tmp_path / 'plots'
    def mutate(prepared, output):
        paths['training'].write_text(paths['training'].read_text() + '\n')
        return {'matplotlib': 'fixture'}
    monkeypatch.setattr(plots, 'render_overview', mutate)
    assert plots.cli(cli_args(paths, out)) == 1
    assert json.loads((out / 'index.json').read_text())['status'] == 'failed'


def test_cli_rejects_nonfinite_json_evidence_before_output(cli_inputs, tmp_path):
    paths, _, _ = cli_inputs
    paths['evaluation'].write_text(paths['evaluation'].read_text()[:-1] + ',"extra":NaN}')
    out = tmp_path / 'plots'
    assert plots.cli(cli_args(paths, out)) == 1
    assert not out.exists()


@pytest.mark.parametrize('extra', [ ['--failure-cases'], ['--segments', 'missing.jsonl'],
    ['--failure-cases', '--segments', 'missing.jsonl', '--splits', 'missing.json'] ])
def test_cli_requires_complete_failure_input_group(cli_inputs, tmp_path, extra):
    paths, _, _ = cli_inputs
    out = tmp_path / 'plots'
    assert plots.cli(cli_args(paths, out) + extra) == 1
    assert not out.exists()


def test_failure_cli_rejects_wrong_metadata_hash_before_audio_decode(cli_inputs, tmp_path, monkeypatch):
    paths, _, _ = cli_inputs
    segments, splits = tmp_path / 'segments.jsonl', tmp_path / 'splits.json'
    segments.write_text('{}\n')
    splits.write_text('{}')
    def forbidden(*args):
        pytest.fail('WAV decoding/validation must not run after input hash mismatch')
    monkeypatch.setattr(plots, 'load_inputs', forbidden)
    out = tmp_path / 'plots'
    assert plots.cli(cli_args(paths, out) + ['--failure-cases', '--segments', str(segments),
        '--splits', str(splits), '--audio-root', str(tmp_path)]) == 1
    assert not out.exists()
