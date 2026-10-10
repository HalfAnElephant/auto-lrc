#!/usr/bin/env python3
"""Train a separate CUDA boundary TCN using train/validation source annotations."""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_coarse_gtsinger import digest, load_inputs, source_errors, summarize, text_key
from scripts.benchmark_dsp_gtsinger import load_coarse_report, read_audio
from t2l.boundary.neural_features import BoundaryConfig, build_candidate_windows, extract_neural_features

TRAINING_DEFAULTS = {
    'seed': 7, 'epochs': 60, 'patience': 10, 'batch_size': 32, 'gradient_accumulation': 2,
    'learning_rate': .0003, 'weight_decay': .01, 'sigma_seconds': .0075,
    'heatmap_pos_weight': 10., 'delta_weight': 1., 'confidence_weight': .1,
    'consistency_weight': .05, 'gradient_clip': 1., 'jitter_seconds': .075,
    'feature_gain_min': .9, 'feature_gain_max': 1.1, 'feature_noise_std': .005,
}
RUNTIME_FILES = ('t2l/boundary/model.py', 't2l/boundary/neural_features.py',
                 't2l/boundary/inference.py', 't2l/boundary/features.py', 'scripts/train_boundary.py',
                 'scripts/benchmark_coarse_gtsinger.py', 'scripts/benchmark_dsp_gtsinger.py',
                 't2l/boundary/augmentation.py')


def runtime_digests():
    return {name: digest(ROOT / name) for name in RUNTIME_FILES}


def write_json(path, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temporary = Path(path).with_suffix('.json.tmp')
    try:
        temporary.write_text(encoded)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path):
    document = json.loads(Path(path).read_text())
    if (not isinstance(document, dict) or set(document) != {'schema_version', 'model', 'training'}
            or document['schema_version'] != 'boundary-training-config-v1'):
        raise ValueError('Expected exact boundary-training-config-v1 fields')
    model = BoundaryConfig.from_dict(document['model'])
    training = document['training']
    if not isinstance(training, dict) or set(training) != set(TRAINING_DEFAULTS):
        raise ValueError('Training config fields differ from the declared contract')
    for name, value in training.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Training value must be finite numeric: ' + name)
    for name in ('seed', 'epochs', 'patience', 'batch_size', 'gradient_accumulation'):
        if type(training[name]) is not int or training[name] < (0 if name == 'seed' else 1):
            raise ValueError('Training value must be a valid integer: ' + name)
    if training['epochs'] > 60 or training['seed'] != 7:
        raise ValueError('This experiment fixes seed=7 and epochs<=60')
    if any(training[key] <= 0 for key in ('learning_rate', 'sigma_seconds', 'heatmap_pos_weight',
                                         'gradient_clip', 'feature_gain_min', 'feature_gain_max')):
        raise ValueError('Positive training hyperparameter required')
    if any(training[key] < 0 for key in ('weight_decay', 'delta_weight', 'confidence_weight',
                                        'consistency_weight', 'jitter_seconds', 'feature_noise_std')):
        raise ValueError('Nonnegative training hyperparameter required')
    if training['feature_gain_min'] > training['feature_gain_max'] or training['jitter_seconds'] > .075:
        raise ValueError('Invalid feature gain range or jitter exceeds 75ms')
    return model, dict(training)


def _references(segment, candidates):
    units = [unit for unit in segment['units'] if unit['kind'] == 'lyric']
    references = {}
    if len(units) != len(candidates):
        return references
    for index, (unit, candidate) in enumerate(zip(units, candidates)):
        if not text_key(unit['text']) or text_key(unit['text']) != text_key(candidate['text']):
            continue
        boundary = unit['boundary']
        sample = boundary.get('candidate_sample')
        if (boundary.get('source') != 'textgrid:first_phone' or type(sample) is not int
                or not 0 <= sample < segment['num_samples']):
            raise ValueError('Invalid published source sample: ' + segment['segment_id'])
        references[index] = sample
    return references


def prepare_training_data(segments, coarse_rows, config):
    """Only train/val enter the decoded cache; test identity checks happen upstream."""
    if not segments or any(segment['split'] not in ('train', 'val') for segment in segments):
        raise ValueError('Training cache accepts train/val only')
    cache, counts = [], Counter()
    for segment in segments:
        candidates = copy.deepcopy(coarse_rows[segment['segment_id']]['runs']['mtl:legacy']['candidates'])
        audio = read_audio(segment)
        features = extract_neural_features(audio, segment['sample_rate'], config=config)
        references = _references(segment, candidates)
        windows = build_candidate_windows(features, candidates, config=config)
        counts[segment['split'] + '_source_boundaries'] += sum(u['kind'] == 'lyric' for u in segment['units'])
        counts[segment['split'] + '_text_matched'] += len(references)
        counts[segment['split'] + '_window_skipped'] += len(windows['skipped'])
        augmented, augmentation = None, None
        if segment['split'] == 'train':
            from t2l.boundary.augmentation import augment_training_audio
            seed = (int.from_bytes(hashlib.sha256(segment['segment_id'].encode()).digest()[:8], 'big') + 7) % (1 << 64)
            augmented_audio, augmentation = augment_training_audio(audio, segment['sample_rate'],
                                                                    np.random.default_rng(seed))
            if (augmented_audio.shape != audio.shape or not np.isfinite(augmented_audio).all()
                    or augmentation.get('source_grid_offset_samples') != 0):
                raise ValueError('Augmentation must preserve the original WAV shape/grid')
            augmentation = dict(augmentation, deterministic_seed=seed)
            augmented = extract_neural_features(augmented_audio, segment['sample_rate'], config=config)
        cache.append({'segment': segment, 'features': features, 'augmented_features': augmented,
                      'augmentation': augmentation, 'candidates': candidates, 'references': references,
                      'audio': audio if segment['split'] == 'val' else None})
    if {entry['segment']['split'] for entry in cache} != {'train', 'val'}:
        raise ValueError('Both train and validation segments are required')
    return cache, dict(counts)


def build_examples(cache, config, training, *, split, rng=None, augment=False):
    """Jitter model centers only, then rebuild original-grid windows and targets."""
    if split not in ('train', 'val') or (augment and (split != 'train' or rng is None)):
        raise ValueError('Augmentation is training-only and requires a seeded generator')
    examples, counts = [], Counter()
    for entry in cache:
        segment = entry['segment']
        if segment['split'] != split:
            continue
        rate = segment['sample_rate']
        candidates = copy.deepcopy(entry['candidates'])
        if augment:
            radius = round(training['jitter_seconds'] * rate)
            for row in candidates:
                start = row.get('start_sample')
                if start is None or row.get('status') == 'unaligned':
                    continue
                end = row.get('end_sample')
                end = min(segment['num_samples'] - 1, end if end is not None else segment['num_samples'] - 1)
                shifted = min(end, max(0, start + int(rng.integers(-radius, radius + 1))))
                row.update(start_sample=shifted, start_seconds=shifted / rate)
        feature_result = entry['features']
        if augment and entry.get('augmented_features') is not None and rng.integers(0, 2):
            feature_result = entry['augmented_features']
            counts['waveform_augmented_segments'] += 1
        else:
            counts['original_waveform_segments'] += 1
        windows = build_candidate_windows(feature_result, candidates, config=config)
        counts['skipped_windows'] += len(windows['skipped'])
        for window_index, candidate_index in enumerate(windows['candidate_indices']):
            if candidate_index not in entry['references']:
                counts['unmatched_windows'] += 1
                continue
            reference = entry['references'][candidate_index]
            positions = windows['sample_positions'][window_index]
            mask = windows['mask'][window_index]
            center = candidates[candidate_index]['start_sample']
            reachable = bool(positions[mask].min() <= reference <= positions[mask].max())
            target = np.exp(-.5 * ((positions - reference) / rate / training['sigma_seconds']) ** 2)
            target = (target * mask).astype(np.float32) if reachable else np.zeros(len(mask), np.float32)
            features = windows['features'][window_index].copy()
            if augment:
                # Feature-domain perturbation; not waveform noise or real accompaniment.
                gain = rng.uniform(training['feature_gain_min'], training['feature_gain_max'])
                features[:49, mask] += 2 * math.log10(gain)
                features[:57, mask] += rng.normal(0, training['feature_noise_std'],
                                                 size=(57, int(mask.sum()))).astype(np.float32)
            examples.append({'features': features, 'phones': windows['phones'][window_index],
                             'mask': mask, 'heatmap': target, 'in_range': reachable,
                             'delta_seconds': (reference - center) / rate,
                             'center_seconds': center / rate, 'segment_id': segment['segment_id'],
                             'candidate_index': candidate_index})
            counts['in_range' if reachable else 'out_of_window'] += 1
    if not examples:
        raise ValueError('No matched usable windows in ' + split)
    return examples, dict(counts)


def tensor_batch(examples, device):
    import torch
    result = {key: torch.from_numpy(np.stack([row[key] for row in examples])).to(device)
              for key in ('features', 'phones', 'mask', 'heatmap')}
    for key in ('in_range', 'delta_seconds', 'center_seconds'):
        result[key] = torch.tensor([row[key] for row in examples], device=device,
                                   dtype=torch.bool if key == 'in_range' else torch.float32)
    result['identities'] = [(row['segment_id'], row['candidate_index']) for row in examples]
    return result


def boundary_loss(outputs, batch, training):
    import torch
    import torch.nn.functional as functional
    logits = outputs['heatmap_logits'].float()
    delta = outputs['delta_seconds'].float()
    confidence = outputs['confidence_logit'].float()
    if (logits.shape != batch['mask'].shape or delta.shape != batch['in_range'].shape
            or confidence.shape != delta.shape or any(not bool(torch.isfinite(v).all())
                                                      for v in (logits, delta, confidence))):
        raise ValueError('Invalid or nonfinite boundary outputs')
    valid = batch['mask'] & batch['in_range'][:, None]
    zero = delta.sum() * 0
    heat = (functional.binary_cross_entropy_with_logits(logits[valid], batch['heatmap'][valid],
             pos_weight=logits.new_tensor(training['heatmap_pos_weight'])) if bool(valid.any()) else zero)
    regression = (functional.smooth_l1_loss(delta[batch['in_range']],
                  batch['delta_seconds'][batch['in_range']], beta=.01) if bool(batch['in_range'].any()) else zero)
    certainty = functional.binary_cross_entropy_with_logits(confidence, batch['in_range'].float())
    predicted = batch['center_seconds'] + delta
    identities = {identity: index for index, identity in enumerate(batch['identities'])}
    pairs = [(index, identities[(segment, position + 1)]) for (segment, position), index in identities.items()
             if (segment, position + 1) in identities]
    consistency = (torch.stack([torch.relu(predicted[left] - predicted[right])
                               for left, right in pairs]).mean() if pairs else zero)
    total = heat + training['delta_weight'] * regression + training['confidence_weight'] * certainty \
            + training['consistency_weight'] * consistency
    return total, {'heatmap': float(heat.detach()), 'delta': float(regression.detach()),
                   'confidence': float(certainty.detach()), 'consistency': float(consistency.detach()),
                   'neighbor_pairs': len(pairs)}


def validate_model(model, config, cache, device):
    from t2l.boundary.inference import refine_boundaries
    runtime = {'model': model, 'config': config, 'device': str(device), 'checkpoint_sha256': None}
    metrics = []
    for entry in cache:
        if entry['segment']['split'] != 'val':
            continue
        predictions = refine_boundaries(entry['audio'], entry['segment']['sample_rate'],
                                        entry['candidates'], loaded=runtime, device=str(device))
        metrics.append(source_errors(entry['segment'], predictions))
    result = summarize(metrics)
    if result['mae_ms'] is None:
        raise ValueError('No evaluated validation boundaries')
    return result


def save_checkpoint(path, model, config, metadata):
    import torch
    state = {key: value.detach().cpu().float() if value.is_floating_point() else value.detach().cpu()
             for key, value in model.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('Nonfinite checkpoint state')
    document = {'schema_version': 'boundary-checkpoint-v1', 'config': config.to_dict(),
                'model_state_dict': state, 'training_metadata': metadata}
    temporary = Path(path).with_suffix('.pt.tmp')
    try:
        torch.save(document, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(cache, config, training, out_dir, report):
    import torch
    from t2l.boundary.model import BoundaryTCN
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires real CUDA with AMP; no CPU fallback')
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    random.seed(training['seed'])
    np.random.seed(training['seed'])
    torch.manual_seed(training['seed'])
    torch.cuda.manual_seed_all(training['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model = BoundaryTCN(config).to(device)
    count = sum(parameter.numel() for parameter in model.parameters())
    if not 500000 <= count <= 2000000:
        raise ValueError('Boundary model must have 0.5M-2M parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
                                 weight_decay=training['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=1024.)
    if not scaler.is_enabled():
        raise RuntimeError('CUDA AMP GradScaler is not active')
    report['environment'] = {'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__),
                             'cuda': torch.version.cuda, 'device': str(device),
                             'torch_num_threads': torch.get_num_threads(),
                             'torch_num_interop_threads': torch.get_num_interop_threads(),
                             'gpu': torch.cuda.get_device_name(device), 'amp': True, 'amp_dtype': 'float16',
                             'amp_initial_scale': 1024.}
    report['parameter_count'] = count
    report['model_parameter_device'] = str(next(model.parameters()).device)
    report['decoded_training_segment_ids'] = [entry['segment']['segment_id'] for entry in cache
                                               if entry['segment']['split'] == 'train']
    report['decoded_validation_segment_ids'] = [entry['segment']['segment_id'] for entry in cache
                                                 if entry['segment']['split'] == 'val']
    report['waveform_augmentation'] = {entry['segment']['segment_id']: entry['augmentation'] for entry in cache
                                       if entry.get('augmentation') is not None}
    _, report['validation_window_counts'] = build_examples(cache, config, training, split='val')
    best, stale = math.inf, 0
    rng = np.random.default_rng(training['seed'])
    started = time.perf_counter()
    cpu_started = time.process_time()
    for epoch in range(1, training['epochs'] + 1):
        epoch_started = time.perf_counter()
        epoch_cpu_started = time.process_time()
        model.train()
        examples, counts = build_examples(cache, config, training, split='train', rng=rng, augment=True)
        order = rng.permutation(len(examples))
        batches = [order[i:i + training['batch_size']] for i in range(0, len(order), training['batch_size'])]
        loss_sum, pairs, steps = 0., 0, 0
        loss_parts = Counter()
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        for number, indices in enumerate(batches):
            batch = tensor_batch([examples[int(index)] for index in indices], device)
            group_start = number // training['gradient_accumulation'] * training['gradient_accumulation']
            group_size = min(training['gradient_accumulation'], len(batches) - group_start)
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                if not torch.is_autocast_enabled():
                    raise RuntimeError('CUDA autocast is not active')
                outputs = model(batch['features'], batch['phones'], batch['mask'])
                if any(value.device != device for value in outputs.values()):
                    raise RuntimeError('Boundary outputs must be on the selected CUDA device')
                loss, parts = boundary_loss(outputs, batch, training)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite training loss')
            scaler.scale(loss / group_size).backward()
            loss_sum += float(loss.detach()) * len(indices)
            pairs += parts['neighbor_pairs']
            for key in ('heatmap', 'delta', 'confidence', 'consistency'):
                loss_parts[key] += parts[key] * len(indices)
            if (number + 1) % training['gradient_accumulation'] == 0 or number + 1 == len(batches):
                scaler.unscale_(optimizer)
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                       for parameter in model.parameters()):
                    raise ValueError('Nonfinite training gradient; no silent AMP skip')
                torch.nn.utils.clip_grad_norm_(model.parameters(), training['gradient_clip'], error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
        model.eval()
        validation = validate_model(model, config, cache, device)
        torch.cuda.synchronize(device)
        history = {'epoch': epoch, 'loss': loss_sum / len(examples), 'train_windows': counts,
                   'loss_parts': {key: value / len(examples) for key, value in loss_parts.items()},
                   'neighbor_pairs_in_batches': pairs, 'optimizer_steps': steps, 'gradient_finite': True,
                   'amp_scale': float(scaler.get_scale()), 'validation': validation,
                   'runtime_seconds': time.perf_counter() - epoch_started,
                   'cpu_seconds': time.process_time() - epoch_cpu_started,
                   'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                   'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
        report['history'].append(history)
        if validation['mae_ms'] < best:
            best, stale = validation['mae_ms'], 0
            metadata = {'input_sha256': report['input_sha256'], 'runtime_sha256': report['runtime_sha256'],
                        'training': training, 'best_epoch': epoch, 'validation': validation,
                        'reference': 'published_dataset_annotation', 'independent_human_accuracy': False,
                        'selection_split': 'val', 'training_splits': ['train', 'val'],
                        'confidence_target': 'source boundary reachable inside original-grid window; uncalibrated'}
            save_checkpoint(out_dir / 'boundary.pt', model, config, metadata)
            report.update(best_epoch=epoch, best_validation=validation)
        else:
            stale += 1
        write_json(out_dir / 'result.json', report)
        print('epoch={} val_mae_ms={:.4f} best_epoch={} stale={}'.format(
            epoch, validation['mae_ms'], report['best_epoch'], stale), flush=True)
        if stale >= training['patience']:
            report['stop_reason'] = 'validation_early_stopping'
            break
    else:
        report['stop_reason'] = 'epoch_limit'
    report['runtime_seconds'] = time.perf_counter() - started
    report['cpu_seconds'] = time.process_time() - cpu_started
    report['runtime_scope'] = 'training epochs and per-epoch validation; excludes input checks and feature cache'
    report['checkpoint'] = {'path': str(out_dir / 'boundary.pt'), 'sha256': digest(out_dir / 'boundary.pt')}


def cli(argv=None):
    cli_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('segments', 'splits', 'audio-root', 'coarse-report', 'out-dir', 'config'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args(argv)
    report, created = None, False
    try:
        config, training = load_config(args.config)
        inputs = {name: digest(path) for name, path in
                  (('segments', args.segments), ('splits', args.splits), ('coarse_report', args.coarse_report),
                   ('config', args.config))}
        segments = load_inputs(args.segments, args.splits, args.audio_root)
        coarse = load_coarse_report(args.coarse_report, segments, inputs)
        args.out_dir.mkdir(parents=True, exist_ok=False)
        created = True
        report = {'schema_version': 1, 'protocol': 'boundary_training', 'status': 'running',
                  'started_at': datetime.now(timezone.utc).isoformat(), 'input_sha256': inputs,
                  'runtime_sha256': runtime_digests(), 'config': config.to_dict(), 'training': training,
                  'reference': 'published_dataset_annotation', 'independent_human_accuracy': False,
                  'training_coarse_method': 'mtl:legacy', 'selection_split': 'val', 'history': [],
                  'limitations': ['No test WAV decode, target, loss or selection; upstream hashes/headers validated.',
                      'Additional feature gain/noise perturbations are distinct from waveform augmentation.',
                      'Training-only synthetic accompaniment and causal tails are not actual instrumental recordings.',
                      'Jitter changes copied coarse window centers only, never frozen predictions or WAV time.',
                      'Out-of-window source targets are masked from heatmap/delta, counted and retained in validation.',
                      'Confidence is uncalibrated window reachability, not probability of onset correctness.',
                      'Adjacent minibatch consistency does not guarantee global order; inference falls back on crossings.',
                      'Published annotations and dataset song identities are not independently human verified.']}
        write_json(args.out_dir / 'result.json', report)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        cache_started, cache_cpu_started = time.perf_counter(), time.process_time()
        cache, report['cache_counts'] = prepare_training_data(
            [segment for segment in segments if segment['split'] in ('train', 'val')], coarse, config)
        report['feature_cache_seconds'] = time.perf_counter() - cache_started
        report['feature_cache_cpu_seconds'] = time.process_time() - cache_cpu_started
        write_json(args.out_dir / 'result.json', report)
        train(cache, config, training, args.out_dir, report)
        if runtime_digests() != report['runtime_sha256'] or any(digest(path) != inputs[name] for name, path in
                (('segments', args.segments), ('splits', args.splits), ('coarse_report', args.coarse_report), ('config', args.config))):
            raise ValueError('Training inputs or runtime changed during execution')
        report['status'] = 'completed'
        report['total_wall_seconds'] = time.perf_counter() - cli_started
        write_json(args.out_dir / 'result.json', report)
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
