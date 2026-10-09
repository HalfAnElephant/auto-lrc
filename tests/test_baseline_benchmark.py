from contextlib import contextmanager
import json
import random

import numpy as np
import pytest
import torch

from scripts import benchmark_baseline as benchmark


def test_layout_ablation_is_instance_local_and_preserves_weights():
    legacy = torch.nn.Sequential(torch.nn.LSTM(2, 3, batch_first=False))
    ablation = torch.nn.Sequential(torch.nn.LSTM(2, 3, batch_first=False))
    before = {key: value.clone() for key, value in ablation.state_dict().items()}

    report = benchmark.configure_lstm((ablation, None), "batch-first")

    assert legacy[0].batch_first is False
    assert ablation[0].batch_first is True
    assert report[0]["state_dict_shapes_unchanged"] is True
    assert report[0]["layers"] == [{"name": "0", "loaded_batch_first": False,
                                     "run_batch_first": True}]
    assert all(torch.equal(before[key], value) for key, value in ablation.state_dict().items())
    benchmark.configure_lstm((legacy, None), "legacy")
    assert legacy[0].batch_first is False


def test_seed_resets_all_random_generators():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        benchmark.reset_seed(7)
        first = (random.random(), np.random.random(), torch.rand(2))
        benchmark.reset_seed(7)
        second = (random.random(), np.random.random(), torch.rand(2))
        assert first[:2] == second[:2]
        assert torch.equal(first[2], second[2])
    finally:
        torch.use_deterministic_algorithms(previous)


def test_repeatability_excludes_runtime_but_includes_raw_boundaries_and_score():
    details = {"lrc": "[00:00.000]word", "units": [{"start_frame": 0}],
               "alignment": {"score": -3.0, "runtime_seconds": 1.0}}
    expected = benchmark.prediction_fingerprint(details)
    details["alignment"]["runtime_seconds"] = 2.0
    assert benchmark.prediction_fingerprint(details) == expected
    details["units"][0]["start_frame"] = 1
    assert benchmark.prediction_fingerprint(details) != expected
    details["units"][0]["start_frame"] = 0
    details["alignment"]["score"] = -4.0
    assert benchmark.prediction_fingerprint(details) != expected


def test_cli_saves_repeatable_raw_results_without_inventing_accuracy(tmp_path, monkeypatch):
    from t2l import t2l
    from t2l.mtl import wrapper

    lyrics, audio = tmp_path / "words.txt", tmp_path / "audio.wav"
    lyrics.write_text("word", encoding="utf-8")
    audio.write_bytes(b"fake audio")
    output = tmp_path / "results"
    layouts = []

    def load_model(method, cuda, verbose):
        assert method == "MTL" and cuda is False
        return (torch.nn.Sequential(torch.nn.LSTM(2, 3, batch_first=False)),
                None, "MTL", False, "cpu")

    def process(lines, audio_path, mtl_model, vocalize, verbose, return_details):
        assert lines == ["word"] and audio_path == str(audio)
        assert return_details is True and vocalize is False
        layouts.append(mtl_model[0][0].batch_first)
        return {"lrc": "[00:00.000]<00:00.000>word", "units": [{"start_frame": 0}],
                "alignment": {"score": float(torch.rand(1))}}

    @contextmanager
    def audit(models):
        yield {"observed_shapes": {}}

    monkeypatch.setattr(wrapper, "load_mtl_model", load_model)
    monkeypatch.setattr(t2l, "process", process)
    monkeypatch.setattr(benchmark, "capture_timeline", audit)
    monkeypatch.setattr(benchmark, "environment", lambda: {})
    monkeypatch.setattr(benchmark, "file_provenance", lambda path: {"path": str(path)})
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        assert benchmark.cli([str(lyrics), str(audio), "-o", str(output), "--variants", "mtl",
                              "--lstm-layouts", "legacy,batch-first", "--repeats", "2"]) == 0
    finally:
        torch.use_deterministic_algorithms(previous)
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["runs"]) == 4
    assert all(row["identical_predictions_and_score"] for row in summary["repeatability"])
    assert layouts == [False, False, True, True]
    assert summary["accuracy_metrics"]["available"] is False
    assert summary["accuracy_metrics"]["mae_seconds"] is None
    assert summary["accuracy_metrics"]["p90_seconds"] is None
    assert summary["accuracy_metrics"]["coverage"] is None
    assert (output / "mtl-legacy-r1.standard.lrc").read_text() == "[00:00.000]word"
    saved = json.loads((output / "mtl-legacy-r1.json").read_text())
    assert saved["result"]["units"] == [{"start_frame": 0}]


def test_parser_defaults_and_explicit_ablation():
    args = benchmark.build_parser().parse_args(["words.txt", "audio.wav", "-o", "results"])
    assert (args.variants, args.lstm_layouts, args.seed, args.threads, args.vocalize) == (
        "mtl,mtl_bdr", "legacy", 7, 2, 0)
    args = benchmark.build_parser().parse_args([
        "words.txt", "audio.wav", "-o", "results", "--lstm-layouts", "legacy,batch-first"])
    assert args.lstm_layouts == "legacy,batch-first"


@pytest.mark.parametrize("option,value", [("--threads", "0"), ("--repeats", "0"),
                                          ("--vocalize", "2")])
def test_parser_rejects_invalid_counts(option, value):
    with pytest.raises(SystemExit):
        benchmark.build_parser().parse_args(["words.txt", "audio.wav", "-o", "results", option, value])


def test_existing_output_is_not_overwritten(tmp_path):
    with pytest.raises(SystemExit):
        benchmark.cli([__file__, __file__, "-o", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_invalid_prediction_still_leaves_readable_failure_report(tmp_path, monkeypatch):
    from t2l import t2l
    from t2l.mtl import wrapper

    lyrics, audio = tmp_path / 'words.txt', tmp_path / 'audio.wav'
    lyrics.write_text('word', encoding='utf-8')
    audio.write_bytes(b'fixture')
    output = tmp_path / 'failed'
    monkeypatch.setattr(wrapper, 'load_mtl_model', lambda *a, **kw: (None, None))
    monkeypatch.setattr(t2l, 'process', lambda *a, **kw: {
        'lrc': '', 'units': [], 'alignment': {'score': float('nan')}})
    monkeypatch.setattr(benchmark, 'environment', lambda: {})
    monkeypatch.setattr(benchmark, 'file_provenance', lambda p: {'path': str(p)})
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda n: None)

    @contextmanager
    def audit(models):
        yield {}

    monkeypatch.setattr(benchmark, 'capture_timeline', audit)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        with pytest.raises(ValueError, match='JSON compliant'):
            benchmark.cli([str(lyrics), str(audio), '-o', str(output), '--variants', 'mtl'])
    finally:
        torch.use_deterministic_algorithms(previous)
    row = json.loads((output / 'mtl-legacy-r1.json').read_text())
    assert row['status'] == 'failed'
    assert row['error']['type'] == 'ValueError'
    assert 'raw_alignment_score' not in row
    assert json.loads((output / 'summary.json').read_text())['runs'] == [row]
    assert not list(output.glob('*.lrc'))


def test_timeline_hooks_capture_real_geometry_and_are_removed():
    from t2l.mtl.model import AcousticModel, train_audio_transforms

    model = AcousticModel(0, 2, 41, 1).eval()
    before = len(train_audio_transforms._forward_hooks)
    with benchmark.capture_timeline((model, None)) as audit, torch.inference_mode():
        features = train_audio_transforms(torch.zeros(1, 3072)).unsqueeze(1)
        output = model(features)
    assert audit["stft"]["hop_length"] == 256
    assert audit["stft"]["center"] is True
    assert audit["observed_shapes"]["mel"]["output"] == [1, 128, 13]
    assert audit["observed_shapes"]["acoustic"]["output"] == list(output.shape) == [1, 4, 41]
    assert audit["observed_shapes"]["acoustic.bilstm.1.BiLSTM"]["interpreted_time_axis"] == 0
    assert audit["nominal_output_step_seconds"] == 768 / 22050
    assert len(train_audio_transforms._forward_hooks) == before
    assert not any(module._forward_hooks for module in model.modules())
