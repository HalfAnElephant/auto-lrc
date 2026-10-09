"""Synthetic geometry checks, not evidence of real singing onset accuracy."""
import julius
import torch

from t2l.mtl.model import AcousticModel, BoundaryDetection, train_audio_transforms


def test_centered_stft_impulse_and_pooling_geometry():
    hop = train_audio_transforms[0].spectrogram.hop_length
    waveform = torch.zeros(1, 17 * hop)
    waveform[0, 8 * hop] = 1
    mel = train_audio_transforms(waveform)
    assert mel.shape == (1, 128, 18)
    assert int(mel.sum(dim=1).argmax()) == 8
    for model_class, classes in ((AcousticModel, (41, 47)), (BoundaryDetection, 1)):
        model = model_class(1, 4, classes, 2).eval()
        with torch.inference_mode():
            output = model(mel.unsqueeze(1))
        assert output.shape[1] == mel.shape[-1] // 3


def test_resampling_preserves_synthetic_impulse_time():
    waveform = torch.zeros(1, 48000)
    waveform[0, 24000] = 1
    resampled = julius.resample_frac(waveform, 48000, 22050)
    assert resampled.shape[-1] == 22050
    assert int(resampled.abs().argmax()) == 11025


def test_legacy_lstm_layout_interprets_batch_as_sequence():
    layer = torch.nn.LSTM(2, 2, bidirectional=True, batch_first=False)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.fill_(0.2)
    inputs = torch.zeros(2, 5, 2)
    perturbed = inputs.clone()
    perturbed[1] = 1
    with torch.inference_mode():
        old, _ = layer(inputs)
        changed, _ = layer(perturbed)
        assert not torch.equal(old[0], changed[0])
        layer.batch_first = True
        fixed, _ = layer(inputs)
        fixed_changed, _ = layer(perturbed)
        assert torch.equal(fixed[0], fixed_changed[0])
