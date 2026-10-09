import json

import numpy as np
import pytest
import torch
from torch import nn

from t2l.mtl import wrapper


class RecordingTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.waveform = None

    def forward(self, waveform):
        self.waveform = waveform.detach().cpu().clone()
        return torch.zeros((waveform.shape[0], 128, 9), device=waveform.device)


class FakeAcousticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inference_mode = False

    def forward(self, features):
        self.inference_mode = torch.is_inference_mode_enabled()
        return torch.zeros((1, 3, 41), device=features.device)


class FakeBoundaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inference_mode = False

    def forward(self, features):
        self.inference_mode = torch.is_inference_mode_enabled()
        return torch.full((1, 3), 0.5, device=features.device)


def test_align_uses_left_channel_and_inference_mode(monkeypatch, capsys):
    transform = RecordingTransform()
    acoustic_model = FakeAcousticModel()
    audio = torch.stack((torch.arange(2304), torch.arange(2304) + 10000)).float()
    captured = {}

    def fake_alignment(song_pred, lyrics, idx):
        captured["shape"] = song_pred.shape
        return [[0, 1]], 0.0

    monkeypatch.setattr(wrapper, "train_audio_transforms", transform)
    monkeypatch.setattr(wrapper.utils, "alignment", fake_alignment)

    word_align, _ = wrapper.align(
        audio,
        ["word"],
        ["AA"],
        np.array([[0, 0]]),
        np.array([[0, 0]]),
        method=(acoustic_model, None, "Baseline", False, "cpu"),
        verbose=False,
    )

    assert word_align == [[0, 1]]
    assert torch.equal(transform.waveform, audio[:1])
    assert acoustic_model.inference_mode
    assert captured["shape"] == (3, 41)
    assert capsys.readouterr().out == ""


def test_align_uses_full_model_output_for_frame_count(monkeypatch):
    transform = RecordingTransform()
    acoustic_model = FakeAcousticModel()
    captured = {}

    def fake_alignment(song_pred, lyrics, idx):
        captured["shape"] = song_pred.shape
        return [[0, 1]], 0.0

    monkeypatch.setattr(wrapper, "train_audio_transforms", transform)
    monkeypatch.setattr(wrapper.utils, "alignment", fake_alignment)

    wrapper.align(
        torch.zeros((1, 1280)),
        ["word"],
        ["AA"],
        np.array([[0, 0]]),
        np.array([[0, 0]]),
        method=(acoustic_model, None, "Baseline", False, "cpu"),
        verbose=False,
    )

    assert captured["shape"] == (3, 41)


def test_align_converts_numpy_audio_to_float32(monkeypatch):
    transform = RecordingTransform()
    acoustic_model = FakeAcousticModel()

    monkeypatch.setattr(wrapper, "train_audio_transforms", transform)
    monkeypatch.setattr(
        wrapper.utils,
        "alignment",
        lambda *args, **kwargs: ([[0, 1]], 0.0),
    )

    wrapper.align(
        np.zeros((1, 2304)),
        ["word"],
        ["AA"],
        np.array([[0, 0]]),
        np.array([[0, 0]]),
        method=(acoustic_model, None, "Baseline", False, "cpu"),
        verbose=False,
    )

    assert transform.waveform.dtype == torch.float32


def test_bdr_alignment_receives_cpu_numpy_predictions(monkeypatch):
    transform = RecordingTransform()
    acoustic_model = FakeAcousticModel()
    boundary_model = FakeBoundaryModel()
    captured = {}

    def fake_alignment_bdr(song_pred, lyrics, idx, bdr_pred, line_start):
        captured["song_pred"] = song_pred
        captured["bdr_pred"] = bdr_pred
        return [[0, 1]], 0.0

    monkeypatch.setattr(wrapper, "train_audio_transforms", transform)
    monkeypatch.setattr(wrapper.utils, "alignment_bdr", fake_alignment_bdr)

    wrapper.align(
        torch.zeros((1, 2304)),
        ["word"],
        ["AA"],
        np.array([[0, 0]]),
        np.array([[0, 0]]),
        method=(acoustic_model, boundary_model, "Baseline", True, "cpu"),
        verbose=False,
    )

    assert boundary_model.inference_mode
    assert isinstance(captured["song_pred"], np.ndarray)
    assert isinstance(captured["bdr_pred"], np.ndarray)


def test_align_accepts_one_dimensional_audio(monkeypatch):
    transform = RecordingTransform()
    acoustic_model = FakeAcousticModel()

    monkeypatch.setattr(wrapper, "train_audio_transforms", transform)
    monkeypatch.setattr(
        wrapper.utils,
        "alignment",
        lambda *args, **kwargs: ([[0, 1]], 0.0),
    )

    wrapper.align(
        torch.arange(2304).float(),
        ["word"],
        ["AA"],
        np.array([[0, 0]]),
        np.array([[0, 0]]),
        method=(acoustic_model, None, "Baseline", False, "cpu"),
        verbose=False,
    )

    assert transform.waveform.shape == (1, 2304)


def test_align_rejects_empty_or_invalid_audio_dimensions():
    model = FakeAcousticModel()
    method = (model, None, "Baseline", False, "cpu")

    for audio in (torch.zeros((0, 10)), torch.zeros((1, 0)), torch.zeros((1, 2, 3))):
        try:
            wrapper.align(
                audio, [], ["AA"], np.array([[0, 0]]), np.array([[0, 0]]),
                method=method, verbose=False,
            )
        except ValueError as error:
            assert "audio" in str(error).lower()
        else:
            raise AssertionError("invalid audio shape was accepted")


@pytest.mark.parametrize('bdr', [False, True])
def test_alignment_details_preserve_legacy_path_and_rng(monkeypatch, bdr):
    monkeypatch.setattr(wrapper, 'train_audio_transforms', RecordingTransform())
    method = (FakeAcousticModel(), FakeBoundaryModel() if bdr else None,
              'Baseline', bdr, 'cpu')
    arguments = (torch.zeros(2304), ['word'], ['AA'], np.array([[0, 1]]),
                 np.array([[0, 1]]))

    torch.manual_seed(23)
    legacy = wrapper.align(*arguments, method=method, verbose=False)
    legacy_rng = torch.get_rng_state()
    torch.manual_seed(23)
    detailed = wrapper.align(*arguments, method=method, verbose=False, return_details=True)

    assert detailed[:2] == legacy
    assert torch.equal(torch.get_rng_state(), legacy_rng)
    details = detailed[2]
    assert details['word_frames'] == legacy[0]
    assert details['phonemes'] == ['AA']
    assert details['word_phoneme_indices'] == [[0, 1]]
    assert details['line_phoneme_indices'] == [[0, 1]]
    assert details['input_samples'] == 2304
    assert details['mel_frames'] == 9
    assert details['output_frames'] == 3
    assert details['sample_rate'] == 22050
    assert details['hop_length'] == 256
    assert details['frame_stride'] == 3
    assert details['frame_resolution_seconds'] == 256 / 22050 * 3
    assert details['method'] == ('Baseline_BDR' if bdr else 'Baseline')
    assert details['model_type'] == 'Baseline'
    assert details['bdr'] is bdr
    assert details['device'] == 'cpu'
    assert isinstance(details['score'], float)
    assert 'uncalibrated' in details['score_kind']
    assert ('with_bdr' in details['score_kind']) is bdr
    assert details['legacy_frame_origin_seconds'] == 0.0
    assert details['physical_origin_seconds'] is None
    assert details['physical_origin_calibrated'] is False
    json.dumps(details, allow_nan=False)


@pytest.mark.parametrize('score', [float('-inf'), float('inf'), float('nan')])
def test_diagnostics_reject_nonfinite_path_score(monkeypatch, score):
    monkeypatch.setattr(wrapper, 'train_audio_transforms', RecordingTransform())
    monkeypatch.setattr(wrapper.utils, 'alignment', lambda *args: ([[0, 1]], score))
    method = (FakeAcousticModel(), None, 'Baseline', False, 'cpu')
    arguments = (torch.zeros(2304), ['word'], ['AA'], np.array([[0, 1]]),
                 np.array([[0, 1]]))

    # Preserve the legacy API, but fail before process()/CLI can write diagnostic
    # artifacts whose JSON would otherwise contain an invalid number.
    assert wrapper.align(*arguments, method=method, verbose=False)[0] == [[0, 1]]
    with pytest.raises(wrapper.utils.AlignmentValueError, match='score is not finite'):
        wrapper.align(*arguments, method=method, verbose=False, return_details=True)
