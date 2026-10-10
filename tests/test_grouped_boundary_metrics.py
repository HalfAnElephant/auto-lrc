import pytest

from t2l.evaluation.metrics import classified_records, grouped_summary, record_summary, song_bootstrap


def records():
    return [{'song_id': 'a', 'status': 'evaluated', 'absolute_error_ms': 1., 'language': 'Chinese'},
            {'song_id': 'a', 'status': 'missing', 'language': 'Chinese'},
            {'song_id': 'b', 'status': 'evaluated', 'absolute_error_ms': 100., 'language': 'English'},
            {'song_id': 'b', 'status': 'text_mismatch', 'language': 'English'}]


def test_denominators_include_all_failed_units():
    summary = record_summary(records())
    assert summary['mae_ms'] == 50.5
    assert summary['coverage_of_all']['5'] == .25
    assert summary['coverage_of_evaluated']['5'] == .5
    assert summary['missing_rate'] == summary['text_mismatch_rate'] == .25
    assert grouped_summary(records(), fields=['language'])['language']['Chinese']['missing'] == 1


def test_song_bootstrap_resamples_clusters_and_is_repeatable():
    first = song_bootstrap(records(), replicates=200)
    assert first == song_bootstrap(records(), replicates=200)
    assert first['intervals']['mae_ms'] == [1., 100.]
    assert first['intervals']['coverage_5ms'] == [0., .5]
    assert first['intervals']['missing_rate'] == [.0, .5]


def test_single_song_cannot_generate_independent_song_interval():
    result = song_bootstrap(records()[:2], replicates=100)
    assert result['status'] == 'insufficient_independent_songs'
    assert result['intervals'] == {}
    with pytest.raises(ValueError):
        song_bootstrap(records(), replicates=True)


def test_classifications_keep_missing_and_mismatched_positions():
    segment = {'song_id': 'a', 'language': 'English', 'sample_rate': 48000, 'units': [
        {'text': 'hello', 'kind': 'lyric', 'boundary': {'source': 'textgrid:first_phone', 'candidate_sample': 1000},
         'classifications': {'first_phoneme': {'value': 'fricative'}}},
        {'text': 'SP', 'kind': 'rest'},
        {'text': 'world', 'kind': 'lyric', 'boundary': {'source': 'textgrid:first_phone', 'candidate_sample': 2000}}]}
    result = classified_records(segment, [{'text': 'hello', 'start_seconds': None},
                                          {'text': 'wrong', 'start_seconds': .1}])
    assert [r['unit_index'] for r in result] == [0, 2]
    assert [r['status'] for r in result] == ['missing', 'text_mismatch']
    assert result[0]['first_phoneme'] == 'fricative'
    assert all(r['status'] == 'text_mismatch' for r in classified_records(segment, []))
    segment['units'][0]['text'] = '!!!'
    assert classified_records(segment, [{'text': '!!!', 'start_seconds': .1},
                                        {'text': 'world', 'start_seconds': .1}])[0]['status'] == 'text_mismatch'
