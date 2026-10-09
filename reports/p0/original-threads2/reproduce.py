"""P0 observational harness; tracked original sources remain untouched."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path('/Users/xcy/Program/auto-lrc')
SNAPSHOT = Path('/tmp/auto-lrc-p0-original-2c475cc').resolve()
OUT = ROOT / 'reports/p0/original-threads2'
COMMIT = '2c475cc'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def provenance(path):
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


if Path.cwd().resolve() != SNAPSHOT:
    raise SystemExit('Run from original snapshot root')
if OUT.exists():
    raise SystemExit('Original output already exists; refusing to overwrite')
tracked = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', COMMIT], cwd=ROOT, text=True).splitlines()
source_check = []
for name in tracked:
    data = subprocess.check_output(['git', 'show', COMMIT + ':' + name], cwd=ROOT)
    target = SNAPSHOT / name
    lfs = data.startswith(b'version https://git-lfs.github.com/spec/v1\n')
    if not lfs and target.read_bytes() != data:
        raise SystemExit('Original snapshot mismatch: ' + name)
    source_check.append({'path': name, 'matches_git_source': None if lfs else True,
                         'lfs_payload': lfs, **provenance(target)})
OUT.mkdir(parents=True)
(OUT / 'reproduce.py').write_bytes(Path(__file__).read_bytes())

import numpy as np
import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
from t2l.t2l import process
from t2l.mtl import utils
assert torch.get_num_threads() == 2
assert torch.get_num_interop_threads() == 1
lyrics = Path('demofile/original_txt.txt').read_text(encoding='utf8').splitlines()
manifest = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'original_commit': subprocess.check_output(['git', 'rev-parse', COMMIT], cwd=ROOT, text=True).strip(),
    'snapshot_root': str(SNAPSHOT), 'source_verification': source_check,
    'command': [sys.executable, 'run_original_p0_threads2.py'],
    'environment': {'python': sys.version, 'platform': platform.platform(),
                    'packages': {dist.metadata['Name']: dist.version for dist in metadata.distributions()}},
    'config': {'seed': 7, 'repeats': 2, 'threads': torch.get_num_threads(),
               'interop_threads': torch.get_num_interop_threads(),
               'deterministic_algorithms': True, 'vocalize': False, 'device': 'cpu'},
    'instrumentation': 'Wrap alignment/alignment_bdr to record returned frames/score and input phonemes; call original process unchanged.',
    'accuracy_metrics': {'available': False, 'reason': 'No independently reviewed reference boundaries were supplied.'},
}
write_json(OUT / 'manifest.json', manifest)
capture = {}
for function in ('alignment', 'alignment_bdr'):
    old = getattr(utils, function)
    def record(*args, _old=old, **kwargs):
        frames, score = _old(*args, **kwargs)
        capture.update(word_frames=[[int(start), int(end)] for start, end in frames],
                       score=float(score), phonemes=list(args[1]),
                       word_phoneme_indices=np.asarray(args[2]).tolist(),
                       output_frames=len(args[0]))
        return frames, score
    setattr(utils, function, record)
summary = {'runs': [], 'repeatability': [], 'accuracy_metrics': manifest['accuracy_metrics']}
for method in ('MTL', 'MTL_BDR'):
    fingerprints = []
    for repeat in (1, 2):
        capture.clear()
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        started = time.perf_counter()
        stem = '{}-r{}'.format(method.lower(), repeat)
        row = {'run': stem, 'method': method, 'repeat': repeat}
        try:
            lrc = process(lyrics, 'demofile/original_track.mp3', mtl_model=method,
                          vocalize=False, verbose=True)
            capture['lrc'] = lrc
            row.update(status='completed', seconds=time.perf_counter() - started,
                       unit_count=len(capture['word_frames']), score=capture['score'])
            fingerprints.append(fingerprint(capture))
            write_json(OUT / (stem + '.json'), {**row, **capture})
            (OUT / (stem + '.lrc')).write_text(lrc, encoding='utf8')
        except Exception as exc:
            row.update(status='failed', seconds=time.perf_counter() - started,
                       error={'type': type(exc).__name__, 'message': str(exc)})
            summary['runs'].append(row)
            write_json(OUT / (stem + '.json'), row)
            write_json(OUT / 'summary.json', summary)
            raise
        summary['runs'].append(row)
        write_json(OUT / 'summary.json', summary)
        print(stem, row['seconds'], row['unit_count'], flush=True)
    summary['repeatability'].append({'method': method, 'repeats': len(fingerprints),
        'identical_predictions_and_score': len(set(fingerprints)) == 1,
        'fingerprints': fingerprints})
    write_json(OUT / 'summary.json', summary)
