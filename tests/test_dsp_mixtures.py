import json
from pathlib import Path

from scripts.benchmark_dsp_mixtures import reused_candidates


def test_native_separated_grid_is_retained_without_reusing_original_rate():
    report_path = Path(__file__).resolve().parents[1] / 'reports/p1/mixtures-cuda-20261010/result.json'
    report = json.loads(report_path.read_text())
    row = report['rows'][0]
    run = row['predictions']['separated']['MTL']
    texts = [u['text'] for u in run['result']['units']]
    candidates = reused_candidates(run, 44100, row['separation']['num_samples'], [''.join(texts)], 'zh')
    assert [c['text'] for c in candidates] == texts
    assert all(c['original_sample_rate'] == 44100 for c in candidates)
    assert candidates[1]['start_seconds'] == run['result']['units'][1]['start_seconds']
    assert candidates[1]['start_sample'] == round(candidates[1]['start_seconds'] * 44100)
    assert all(c['confidence'] is None for c in candidates)
