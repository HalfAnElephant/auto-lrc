import copy
import json
import math
import wave

import numpy as np
import pytest
import torch

from scripts import evaluate_boundary as evaluator
from scripts import train_boundary as trainer
from t2l.boundary import inference
from t2l.boundary.model import BoundaryTCN
from t2l.boundary.neural_features import BoundaryConfig


@pytest.fixture(autouse=True, scope='module')
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def fake_features(audio, rate, *, config):
    positions = np.arange(0, len(audio), round(rate * config.hop_seconds), dtype=np.int64)
    return {'sample_positions': positions, 'features': np.zeros((57, len(positions)), np.float32),
            'metadata': {'config': config.to_dict(), 'sample_rate': rate, 'num_samples': len(audio),
                         'hop_samples': round(rate * config.hop_seconds)}}


def candidate(start=24000, *, end=48000, length=48000, text='one'):
    return {'text': text, 'status': 'candidate', 'start_sample': start,
            'start_seconds': start / 48000 if start is not None else None,
            'end_sample': end, 'end_seconds': end / 48000 if end is not None else None,
            'original_sample_rate': 48000, 'original_num_samples': length,
            'phonemes': ['W', 'AH1', 'N'], 'raw_frames': {'start': 7, 'end': 12}, 'confidence': None}


def segment(split='train', *, reference=24123, length=48000):
    return {'segment_id': split, 'song_id': 'song-' + split, 'split': split, 'language': 'English',
            'singer': 'singer-' + split, 'technique': 'Control', 'group': 'Control_Group',
            'sample_rate': 48000, 'num_samples': length, 'units': [{'kind': 'lyric', 'text': 'one',
                'boundary': {'source': 'textgrid:first_phone', 'candidate_sample': reference}}]}


def cached(*, reference=24123, start=24000, end=48000, split='train'):
    config = BoundaryConfig()
    return [{'segment': segment(split, reference=reference), 'features': fake_features(np.zeros(48000), 48000, config=config),
             'candidates': [candidate(start, end=end)], 'references': {0: reference}}]


@pytest.fixture
def dataset(tmp_path):
    audio_root = tmp_path / 'audio'
    segments, assignments, coarse_rows = [], [], []
    for index, split in enumerate(('train', 'val', 'test')):
        row = segment(split)
        path = audio_root / (row['segment_id'] + '.wav')
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            waveform = np.zeros(48000, dtype='<i2')
            waveform[0] = index
            wav.writeframes(waveform.tobytes())
        row.update(audio_path=str(path), sources={'wav': {'sha256': trainer.digest(path)}})
        segments.append(row)
        assignments.append({'segment_id': split, 'canonical_song_id': row['song_id'], 'split': split})
        coarse_rows.append({'segment_id': split, 'song_id': row['song_id'], 'split': split,
                            'runs': {method + ':legacy': {'status': 'completed', 'candidates': [candidate()]}
                                     for method in evaluator.METHODS}})
    paths = {name: tmp_path / (name + extension) for name, extension in
             (('segments', '.jsonl'), ('splits', '.json'), ('coarse', '.json'), ('config', '.json'))}
    config = json.loads((trainer.ROOT / 'configs/boundary-tcn.json').read_text())
    config['training'].update(epochs=2, patience=1)
    paths['config'].write_text(json.dumps(config))
    def save():
        paths['segments'].write_text(''.join(json.dumps(row) + '\n' for row in segments))
        paths['splits'].write_text(json.dumps({'identity_mode': 'dataset', 'held_out_singers': [], 'assignments': assignments}))
        paths['coarse'].write_text(json.dumps({'schema_version': 1, 'status': 'completed',
            'input_sha256': {str(paths['segments']): trainer.digest(paths['segments']),
                             str(paths['splits']): trainer.digest(paths['splits'])}, 'rows': coarse_rows}))
    save()
    return dict(paths=paths, segments=segments, coarse_rows=coarse_rows, audio_root=audio_root,
                save=save, root=tmp_path)


def train_args(dataset, out):
    paths = dataset['paths']
    return ['--segments', str(paths['segments']), '--splits', str(paths['splits']),
            '--audio-root', str(dataset['audio_root']), '--coarse-report', str(paths['coarse']),
            '--config', str(paths['config']), '--out-dir', str(out)]


def evaluate_args(dataset, checkpoint, out):
    paths = dataset['paths']
    return ['--segments', str(paths['segments']), '--splits', str(paths['splits']),
            '--audio-root', str(dataset['audio_root']), '--coarse-report', str(paths['coarse']),
            '--checkpoint', str(checkpoint), '--out-dir', str(out), '--device', 'cpu']


def training_inputs(dataset):
    return {name: trainer.digest(dataset['paths'][path]) for name, path in
            (('segments', 'segments'), ('splits', 'splits'), ('coarse_report', 'coarse'), ('config', 'config'))}


def create_frozen_checkpoint(dataset):
    out = dataset['root'] / 'training'
    out.mkdir()
    config, training = trainer.load_config(dataset['paths']['config'])
    metadata = {'input_sha256': training_inputs(dataset), 'runtime_sha256': trainer.runtime_digests(),
                'training': training, 'best_epoch': 1, 'validation': {'mae_ms': 10.},
                'reference': 'published_dataset_annotation', 'independent_human_accuracy': False,
                'selection_split': 'val', 'training_splits': ['train', 'val']}
    trainer.save_checkpoint(out / 'boundary.pt', torch.nn.Linear(1, 1), config, metadata)
    report = {'schema_version': 1, 'protocol': 'boundary_training', 'status': 'completed',
              'selection_split': 'val', 'input_sha256': metadata['input_sha256'],
              'runtime_sha256': metadata['runtime_sha256'], 'config': config.to_dict(),
              'training': training, 'best_epoch': 1, 'best_validation': metadata['validation'],
              'checkpoint': {'sha256': trainer.digest(out / 'boundary.pt')}}
    trainer.write_json(out / 'result.json', report)
    return out / 'boundary.pt', metadata


def test_declared_configuration_is_reproducible_and_matches_feature_contract():
    config, training = trainer.load_config(trainer.ROOT / 'configs/boundary-tcn.json')
    assert config == BoundaryConfig()
    assert training == trainer.TRAINING_DEFAULTS
    assert (training['seed'], training['epochs'], training['patience']) == (7, 60, 10)
    assert (training['batch_size'], training['gradient_accumulation'], training['learning_rate']) == (32, 2, 3e-4)


@pytest.mark.parametrize('name,value', [('seed', 8), ('seed', True), ('epochs', 61), ('epochs', 1.),
    ('patience', 0), ('learning_rate', 0), ('sigma_seconds', math.nan), ('delta_weight', -1),
    ('jitter_seconds', .076), ('feature_gain_min', 2)])
def test_invalid_training_hyperparameters_rejected(dataset, name, value):
    path = dataset['paths']['config']
    document = json.loads(path.read_text())
    document['training'][name] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        trainer.load_config(path)


@pytest.mark.parametrize('section', ['top', 'training', 'model'])
def test_configuration_schema_rejects_unknown_fields(dataset, section):
    path = dataset['paths']['config']
    document = json.loads(path.read_text())
    (document if section == 'top' else document[section])['unknown'] = 1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        trainer.load_config(path)


def test_training_cache_refuses_test_inputs_before_decode(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('test WAV must not be decoded')
    monkeypatch.setattr(trainer, 'read_audio', forbidden)
    with pytest.raises(ValueError, match='train/val only'):
        trainer.prepare_training_data([segment('test')], {}, BoundaryConfig())


def test_training_cli_does_not_decode_or_read_test_source_labels(dataset, monkeypatch):
    dataset['segments'][2]['units'] = [{'kind': 'lyric', 'text': 'one', 'boundary': {'poison': 'test label'}}]
    dataset['save']()
    decoded, augmentations = [], []
    original_read = trainer.read_audio
    def tracked_read(row):
        decoded.append(row['split'])
        return original_read(row)
    def augmentation(audio, rate, rng):
        augmentations.append(len(audio))
        return audio.astype(np.float32), {'source_grid_offset_samples': 0, 'synthetic': True}
    from t2l.boundary import augmentation as module
    monkeypatch.setattr(module, 'augment_training_audio', augmentation)
    monkeypatch.setattr(trainer, 'read_audio', tracked_read)
    monkeypatch.setattr(trainer, 'extract_neural_features', fake_features)
    def fake_train(cache, config, training, out, report):
        assert [entry['segment']['split'] for entry in cache] == ['train', 'val']
        assert cache[0]['augmented_features'] is not None and cache[1]['augmented_features'] is None
        assert cache[0]['audio'] is None and cache[1]['audio'].shape == (48000,)
        report['best_epoch'] = 1
    monkeypatch.setattr(trainer, 'train', fake_train)
    out = dataset['root'] / 'run'
    assert trainer.cli(train_args(dataset, out)) == 0
    assert decoded == ['train', 'val'] and augmentations == [48000]
    report = json.loads((out / 'result.json').read_text())
    assert report['selection_split'] == 'val' and report['independent_human_accuracy'] is False
    assert report['feature_cache_seconds'] >= 0 and report['total_wall_seconds'] >= report['feature_cache_seconds']


def test_jitter_rebuilds_coarse_coordinates_not_source_times_and_is_seeded():
    cache = cached(reference=24123)
    original = copy.deepcopy(cache[0]['candidates'])
    first, counts = trainer.build_examples(cache, BoundaryConfig(), trainer.TRAINING_DEFAULTS,
                                            split='train', rng=np.random.default_rng(7), augment=True)
    second, _ = trainer.build_examples(cache, BoundaryConfig(), trainer.TRAINING_DEFAULTS,
                                       split='train', rng=np.random.default_rng(7), augment=True)
    example = first[0]
    center = round(example['center_seconds'] * 48000)
    assert center != 24000 and abs(center - 24000) <= 3600
    assert example['delta_seconds'] == pytest.approx((24123 - center) / 48000)
    np.testing.assert_array_equal(example['features'], second[0]['features'])
    assert cache[0]['candidates'] == original and cache[0]['references'] == {0: 24123}
    assert example['in_range'] and counts['in_range'] == 1
    assert np.all(np.abs(example['features'][57]) <= 1)


def test_label_changes_only_targets_never_training_feature_window():
    a, _ = trainer.build_examples(cached(reference=24123), BoundaryConfig(), trainer.TRAINING_DEFAULTS, split='train')
    b, _ = trainer.build_examples(cached(reference=25000), BoundaryConfig(), trainer.TRAINING_DEFAULTS, split='train')
    np.testing.assert_array_equal(a[0]['features'], b[0]['features'])
    np.testing.assert_array_equal(a[0]['phones'], b[0]['phones'])
    assert not np.array_equal(a[0]['heatmap'], b[0]['heatmap'])


@pytest.mark.parametrize('reference,end', [(47000, 48000), (24500, 24200)])
def test_outside_radius_or_end_mask_is_counted_not_faked_at_center(reference, end):
    examples, counts = trainer.build_examples(cached(reference=reference, end=end), BoundaryConfig(),
                                              trainer.TRAINING_DEFAULTS, split='train')
    assert examples[0]['in_range'] is False
    assert np.count_nonzero(examples[0]['heatmap']) == 0
    assert examples[0]['delta_seconds'] == pytest.approx((reference - 24000) / 48000)
    assert counts['out_of_window'] == 1


def test_waveform_augmented_features_selected_only_for_training_and_seeded():
    cache = cached()
    cache[0]['augmented_features'] = copy.deepcopy(cache[0]['features'])
    cache[0]['augmented_features']['features'][:] = 3
    config = dict(trainer.TRAINING_DEFAULTS, jitter_seconds=0, feature_noise_std=0,
                  feature_gain_min=1, feature_gain_max=1)
    used = set()
    for seed in range(10):
        rows, counts = trainer.build_examples(cache, BoundaryConfig(), config, split='train',
                                              rng=np.random.default_rng(seed), augment=True)
        used.add(float(rows[0]['features'][0, 140]))
        assert sum(counts.get(key, 0) for key in ('original_waveform_segments', 'waveform_augmented_segments')) == 1
    assert used == {0., 3.}
    cache[0]['segment']['split'] = 'val'
    rows, _ = trainer.build_examples(cache, BoundaryConfig(), config, split='val')
    assert rows[0]['features'][0, 140] == 0
    with pytest.raises(ValueError, match='training-only'):
        trainer.build_examples(cache, BoundaryConfig(), config, split='val', rng=np.random.default_rng(7), augment=True)


def test_tensor_batch_types_and_soft_target_remain_original_sample_based():
    examples, _ = trainer.build_examples(cached(), BoundaryConfig(), trainer.TRAINING_DEFAULTS, split='train')
    batch = trainer.tensor_batch(examples, 'cpu')
    assert batch['features'].shape == (1, 58, 281) and batch['features'].dtype == torch.float32
    assert batch['mask'].dtype == batch['in_range'].dtype == torch.bool
    assert batch['phones'].dtype == torch.int64
    assert batch['heatmap'].max().item() < 1
    assert batch['delta_seconds'].item() == pytest.approx(123 / 48000)


def test_out_of_window_examples_train_confidence_only_with_finite_backward():
    examples, _ = trainer.build_examples(cached(reference=47000), BoundaryConfig(), trainer.TRAINING_DEFAULTS, split='train')
    batch = trainer.tensor_batch(examples, 'cpu')
    outputs = {'heatmap_logits': torch.zeros(1, 281, requires_grad=True),
               'delta_seconds': torch.zeros(1, requires_grad=True), 'confidence_logit': torch.zeros(1, requires_grad=True)}
    loss, parts = trainer.boundary_loss(outputs, batch, trainer.TRAINING_DEFAULTS)
    assert parts['heatmap'] == parts['delta'] == 0 and parts['confidence'] > 0
    loss.backward()
    assert outputs['heatmap_logits'].grad is None
    assert outputs['confidence_logit'].grad.item() > 0 and torch.isfinite(loss)


def test_small_real_model_all_objectives_have_finite_gradients():
    config = BoundaryConfig(channels=8, norm_groups=2, dropout=0.)
    examples, _ = trainer.build_examples(cached(), BoundaryConfig(), trainer.TRAINING_DEFAULTS, split='train')
    batch = trainer.tensor_batch(examples, 'cpu')
    model = BoundaryTCN(config)
    outputs = model(batch['features'], batch['phones'], batch['mask'])
    loss, parts = trainer.boundary_loss(outputs, batch, trainer.TRAINING_DEFAULTS)
    loss.backward()
    assert parts['heatmap'] > 0 and parts['confidence'] > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_consistency_penalty_only_adjacent_same_segment_and_no_minimum_gap():
    examples, _ = trainer.build_examples(cached(), BoundaryConfig(), trainer.TRAINING_DEFAULTS, split='train')
    rows = [copy.deepcopy(examples[0]) for _ in range(4)]
    for row, identity in zip(rows, [('a', 0), ('a', 1), ('b', 0), ('a', 3)]):
        row['segment_id'], row['candidate_index'] = identity
    batch = trainer.tensor_batch(rows, 'cpu')
    outputs = {'heatmap_logits': torch.zeros(4, 281), 'delta_seconds': torch.tensor([.1, 0., -.3, -.3]),
               'confidence_logit': torch.zeros(4)}
    _, parts = trainer.boundary_loss(outputs, batch, trainer.TRAINING_DEFAULTS)
    assert parts['neighbor_pairs'] == 1 and parts['consistency'] == pytest.approx(.1)
    outputs['delta_seconds'][:] = 0
    _, parts = trainer.boundary_loss(outputs, batch, trainer.TRAINING_DEFAULTS)
    assert parts['consistency'] == 0


@pytest.mark.parametrize('head', ['heatmap_logits', 'delta_seconds', 'confidence_logit'])
def test_nonfinite_model_loss_is_rejected(head):
    examples, _ = trainer.build_examples(cached(), BoundaryConfig(), trainer.TRAINING_DEFAULTS, split='train')
    batch = trainer.tensor_batch(examples, 'cpu')
    outputs = {'heatmap_logits': torch.zeros(1, 281), 'delta_seconds': torch.zeros(1), 'confidence_logit': torch.zeros(1)}
    outputs[head].reshape(-1)[0] = math.nan
    with pytest.raises(ValueError, match='nonfinite'):
        trainer.boundary_loss(outputs, batch, trainer.TRAINING_DEFAULTS)


def test_validation_uses_inference_argmax_fallback_and_all_label_denominators(monkeypatch):
    cache = cached(split='val', reference=47000)
    cache[0]['audio'] = np.zeros(48000)
    calls = []
    def refine(audio, rate, candidates, *, loaded, device):
        calls.append(copy.deepcopy(candidates))
        return copy.deepcopy(candidates)
    monkeypatch.setattr(inference, 'refine_boundaries', refine)
    result = trainer.validate_model(object(), BoundaryConfig(), cache, 'cpu')
    assert result['total'] == result['evaluated'] == 1 and result['mae_ms'] == pytest.approx(23000 / 48)
    assert calls[0] == cache[0]['candidates'] and 'reference_sample' not in calls[0][0]


def test_checkpoint_saves_float32_finite_state_without_copying_source_labels(tmp_path):
    model = torch.nn.Linear(2, 1).half()
    path = tmp_path / 'boundary.pt'
    metadata = {'best_epoch': 1, 'reference': 'published_dataset_annotation'}
    trainer.save_checkpoint(path, model, BoundaryConfig(), metadata)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint['schema_version'] == 'boundary-checkpoint-v1'
    assert all(value.dtype == torch.float32 for value in checkpoint['model_state_dict'].values())
    assert checkpoint['training_metadata'] == metadata
    assert set(checkpoint) == {'schema_version', 'config', 'model_state_dict', 'training_metadata'}
    assert not list(tmp_path.glob('*.tmp'))


def test_nonfinite_checkpoint_state_refuses_publication(tmp_path):
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight[:] = math.nan
    with pytest.raises(ValueError, match='Nonfinite checkpoint'):
        trainer.save_checkpoint(tmp_path / 'boundary.pt', model, BoundaryConfig(), {})
    assert not list(tmp_path.iterdir())


def test_training_requires_real_cuda_not_cpu_mock_success(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='real CUDA'):
        trainer.train([], BoundaryConfig(), trainer.TRAINING_DEFAULTS, tmp_path, {})


def test_failed_training_is_serialized_and_output_is_exclusive(dataset, monkeypatch):
    monkeypatch.setattr(trainer, 'prepare_training_data', lambda *args: ([], {}))
    def fail(*args):
        raise RuntimeError('AMP failed')
    monkeypatch.setattr(trainer, 'train', fail)
    out = dataset['root'] / 'failed'
    assert trainer.cli(train_args(dataset, out)) == 1
    content = (out / 'result.json').read_bytes()
    report = json.loads(content)
    assert report['status'] == 'failed' and report['error'] == {'type': 'RuntimeError', 'message': 'AMP failed'}
    assert report['total_wall_seconds'] >= report['feature_cache_seconds'] >= 0
    assert trainer.cli(train_args(dataset, out)) == 1
    assert (out / 'result.json').read_bytes() == content


def test_json_failure_does_not_publish_nan_or_leave_temporary_file(tmp_path):
    path = tmp_path / 'result.json'
    trainer.write_json(path, {'status': 'first'})
    with pytest.raises(ValueError):
        trainer.write_json(path, {'invalid': math.nan})
    assert json.loads(path.read_text()) == {'status': 'first'} and not list(tmp_path.glob('*.tmp'))


def test_frozen_checkpoint_binds_completed_training_and_all_input_hashes(dataset):
    path, metadata = create_frozen_checkpoint(dataset)
    inputs = training_inputs(dataset)
    assert evaluator.verify_training_checkpoint(path, inputs) == metadata
    inputs['segments'] = '0' * 64
    with pytest.raises(ValueError, match='Evaluation input SHA256'):
        evaluator.verify_training_checkpoint(path, inputs)


@pytest.mark.parametrize('change', ['status', 'checkpoint_hash', 'runtime', 'config', 'epoch', 'validation', 'selection'])
def test_changed_or_failed_training_evidence_rejected(dataset, change):
    path, _ = create_frozen_checkpoint(dataset)
    report_path = path.parent / 'result.json'
    report = json.loads(report_path.read_text())
    if change == 'status':
        report['status'] = 'failed'
    elif change == 'checkpoint_hash':
        report['checkpoint']['sha256'] = '0' * 64
    elif change == 'runtime':
        report['runtime_sha256']['t2l/boundary/inference.py'] = '0' * 64
    elif change == 'config':
        report['config']['channels'] = 64
    elif change == 'epoch':
        report['best_epoch'] = 2
    elif change == 'validation':
        report['best_validation']['mae_ms'] = 0
    else:
        report['selection_split'] = 'test'
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        evaluator.verify_training_checkpoint(path, training_inputs(dataset))


@pytest.fixture
def fake_refinement(monkeypatch):
    calls = []
    def dsp(audio, rate, candidates, *, config, features):
        calls.append(('dsp', config.to_dict(), copy.deepcopy(candidates)))
        result = copy.deepcopy(candidates)
        for row in result:
            if row['start_sample'] is not None:
                row.update(start_sample=24060, start_seconds=24060 / rate,
                           refinement={'status': 'refined', 'reason': None})
        return result
    def neural(audio, rate, candidates, *, loaded, device, batch_size):
        calls.append(('neural', batch_size, copy.deepcopy(candidates)))
        result = copy.deepcopy(candidates)
        for row in result:
            if row['start_sample'] is not None:
                row.update(start_sample=24120, start_seconds=24120 / rate,
                           refinement={'status': 'refined', 'reason': None})
        return result
    monkeypatch.setattr(evaluator, 'refine_dsp', dsp)
    monkeypatch.setattr(inference, 'refine_boundaries', neural)
    monkeypatch.setattr(evaluator, 'extract_features', lambda *args, **kwargs: {'metadata': {'resampled': False}})
    return calls


def test_evaluation_has_fixed_four_variants_two_models_and_never_sends_source_labels(dataset, fake_refinement):
    by_id = {row['segment_id']: row for row in dataset['coarse_rows']}
    result = evaluator.evaluate(dataset['segments'], by_id, {'device': 'cpu'})
    assert result['failure_count'] == 0
    assert set(result['rows'][0]['runs']['mtl']) == set(evaluator.VARIANTS)
    assert set(key.split(':')[1] for key in result['summary']) == set(evaluator.VARIANTS)
    assert len([call for call in fake_refinement if call[0] == 'dsp']) == 6
    assert len([call for call in fake_refinement if call[0] == 'neural']) == 12
    assert all(call[1] == evaluator.DSP_TRIAL.to_dict() for call in fake_refinement if call[0] == 'dsp')
    assert all('boundary' not in row and 'reference_sample' not in row for call in fake_refinement for row in call[2])
    assert set(result['song_bootstrap']) == {method + ':' + variant + ':test'
                                          for method in evaluator.METHODS for variant in evaluator.VARIANTS}
    assert result['rows'][0]['runs']['mtl']['dsp_neural']['dsp_candidates'][0]['start_sample'] == 24060


def test_evaluation_retains_missing_and_text_mismatch_denominators(dataset, fake_refinement):
    by_id = copy.deepcopy({row['segment_id']: row for row in dataset['coarse_rows']})
    by_id['val']['runs']['mtl:legacy']['candidates'][0].update(start_sample=None, start_seconds=None, status='unaligned')
    by_id['test']['runs']['mtl:legacy']['candidates'][0]['text'] = 'wrong'
    result = evaluator.evaluate(dataset['segments'], by_id, {'device': 'cpu'})
    for variant in evaluator.VARIANTS:
        val = result['summary']['mtl:' + variant + ':val']
        test = result['summary']['mtl:' + variant + ':test']
        assert val['total'] == val['missing'] == 1 and val['evaluated'] == 0
        assert test['total'] == test['text_mismatch'] == 1 and test['evaluated'] == 0
        assert val['coverage_of_all']['5'] == test['coverage_of_all']['5'] == 0


def test_neural_failure_is_explicit_and_every_label_stays_in_denominator(dataset, fake_refinement, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('device fault')
    monkeypatch.setattr(inference, 'refine_boundaries', fail)
    result = evaluator.evaluate(dataset['segments'], {row['segment_id']: row for row in dataset['coarse_rows']}, {'device': 'cpu'})
    assert result['failure_count'] == 12 and len(result['failure_examples']) == 12
    for key, summary in result['summary'].items():
        if key.split(':')[1] in ('neural', 'dsp_neural'):
            assert summary['total'] == summary['missing'] == 1 and summary['evaluated'] == 0
    assert result['rows'][0]['runs']['mtl']['coarse']['status'] == 'completed'


def test_dsp_failure_does_not_prevent_standalone_neural_and_is_not_silently_replaced(dataset, fake_refinement, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('bad DSP')
    monkeypatch.setattr(evaluator, 'refine_dsp', fail)
    result = evaluator.evaluate(dataset['segments'], {row['segment_id']: row for row in dataset['coarse_rows']}, {'device': 'cpu'})
    assert result['failure_count'] == 12
    for row in result['rows']:
        for runs in row['runs'].values():
            assert runs['neural']['status'] == 'completed'
            assert runs['dsp']['status'] == runs['dsp_neural']['status'] == 'failed'


def test_test_label_changes_metrics_but_cannot_change_refinement_calls(dataset, fake_refinement):
    by_id = {row['segment_id']: row for row in dataset['coarse_rows']}
    first = evaluator.evaluate(dataset['segments'], by_id, {'device': 'cpu'})
    first_calls = copy.deepcopy(fake_refinement)
    fake_refinement.clear()
    dataset['segments'][2]['units'][0]['boundary']['candidate_sample'] = 47000
    second = evaluator.evaluate(dataset['segments'], by_id, {'device': 'cpu'})
    assert first_calls == fake_refinement
    assert first['summary']['mtl:neural:val'] == second['summary']['mtl:neural:val']
    assert first['summary']['mtl:neural:test'] != second['summary']['mtl:neural:test']


def test_evaluation_cli_leaves_frozen_selection_and_checkpoint_bytes_unchanged(dataset, fake_refinement, monkeypatch):
    path, _ = create_frozen_checkpoint(dataset)
    original = (path.read_bytes(), (path.parent / 'result.json').read_bytes())
    monkeypatch.setattr(inference, 'load_boundary_checkpoint', lambda *args, **kwargs: {'device': 'cpu', 'config': BoundaryConfig()})
    out = dataset['root'] / 'evaluation'
    assert evaluator.cli(evaluate_args(dataset, path, out)) == 0
    assert (path.read_bytes(), (path.parent / 'result.json').read_bytes()) == original
    report = json.loads((out / 'result.json').read_text())
    assert report['status'] == 'completed' and report['test_used_for_selection'] is False
    assert report['total_wall_seconds'] >= report['runtime_seconds'] >= 0
    assert report['fixed_dsp_trial_config'] == evaluator.DSP_TRIAL.to_dict()
    before = (out / 'result.json').read_bytes()
    assert evaluator.cli(evaluate_args(dataset, path, out)) == 1
    assert (out / 'result.json').read_bytes() == before


def test_evaluation_cli_reports_runtime_failure_not_completed(dataset, fake_refinement, monkeypatch):
    path, _ = create_frozen_checkpoint(dataset)
    monkeypatch.setattr(inference, 'load_boundary_checkpoint', lambda *args, **kwargs: {'device': 'cpu', 'config': BoundaryConfig()})
    def fail(*args, **kwargs):
        raise RuntimeError('fault')
    monkeypatch.setattr(inference, 'refine_boundaries', fail)
    out = dataset['root'] / 'evaluation'
    assert evaluator.cli(evaluate_args(dataset, path, out)) == 1
    report = json.loads((out / 'result.json').read_text())
    assert report['status'] == 'failed' and report['failure_count'] == 12


def test_input_hash_mismatch_fails_before_evaluation_or_output(dataset, monkeypatch):
    path, _ = create_frozen_checkpoint(dataset)
    dataset['segments'][2]['units'][0]['boundary']['candidate_sample'] += 1
    dataset['save']()
    def forbidden(*args, **kwargs):
        raise AssertionError('changed dataset must not be evaluated')
    monkeypatch.setattr(evaluator, 'evaluate', forbidden)
    out = dataset['root'] / 'changed'
    assert evaluator.cli(evaluate_args(dataset, path, out)) == 1 and not out.exists()
