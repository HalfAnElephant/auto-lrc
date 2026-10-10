import copy

import pytest

from scripts.cache_training_coarse import reusable_runs


def fixture():
    segment = {'segment_id': 'song/a', 'song_id': 'song', 'split': 'train', 'language': 'English',
               'sample_rate': 48000, 'num_samples': 48000, 'sources': {'wav': {'sha256': 'abc'}},
               'units': [{'kind': 'lyric', 'text': 'word', 'boundary': {'source': 'textgrid:first_phone',
                                                                     'candidate_sample': 4800}}]}
    candidate = {'text': 'word', 'start_seconds': .12, 'original_sample_rate': 48000,
                 'original_num_samples': 48000, 'status': 'candidate'}
    row = {'segment_id': 'song/a', 'song_id': 'song', 'split': 'train', 'runs': {
        method + ':legacy': {'status': 'completed', 'candidates': [copy.deepcopy(candidate)],
                            'observed_forwards': [{'device': 'cuda:0'}]} for method in ('mtl', 'mtl_bdr')}}
    return segment, row


def test_cache_reuses_only_predictions_and_recomputes_source_errors():
    segment, row = fixture()
    previous = copy.deepcopy(segment)
    previous['units'][0]['boundary']['candidate_sample'] = 6000
    runs = reusable_runs(segment, previous, row)
    assert runs['mtl:legacy']['source_metrics']['errors'][0]['signed_error_ms'] == pytest.approx(20)
    assert runs['mtl:legacy']['observed_forwards'] == [{'device': 'cuda:0'}]
    runs['mtl:legacy']['candidates'][0]['start_seconds'] = .5
    assert row['runs']['mtl:legacy']['candidates'][0]['start_seconds'] == .12


@pytest.mark.parametrize('field,value', [('split', 'test'), ('song_id', 'another'),
                                        ('sample_rate', 44100), ('num_samples', 48001),
                                        ('language', 'Chinese')])
def test_cache_rejects_changed_inputs(field, value):
    segment, row = fixture()
    changed = copy.deepcopy(segment)
    changed[field] = value
    with pytest.raises(ValueError, match='input differs'):
        reusable_runs(segment, changed, row)


def test_cache_rejects_audio_or_lyrics_change():
    segment, row = fixture()
    changed = copy.deepcopy(segment)
    changed['sources']['wav']['sha256'] = 'other'
    with pytest.raises(ValueError):
        reusable_runs(segment, changed, row)
    changed = copy.deepcopy(segment)
    changed['units'][0]['text'] = 'world'
    with pytest.raises(ValueError):
        reusable_runs(segment, changed, row)


def test_cache_rejects_failed_or_wrong_sample_grid():
    segment, row = fixture()
    row['runs']['mtl:legacy']['status'] = 'failed'
    with pytest.raises(ValueError, match='not completed'):
        reusable_runs(segment, segment, row)
    row['runs']['mtl:legacy']['status'] = 'completed'
    row['runs']['mtl:legacy']['candidates'][0]['original_sample_rate'] = 22050
    with pytest.raises(ValueError, match='sample grid'):
        reusable_runs(segment, segment, row)
