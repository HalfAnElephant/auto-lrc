"""Opt-in decoder integration and original-audio metadata; no checkpoints."""
import numpy as np
import pytest
import torch
from torch import nn
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from t2l.mtl import wrapper
from t2l import t2l as pipeline


class Transform(nn.Module):
    def forward(self, x):
        return torch.zeros((1, 128, 18))


class Acoustic(nn.Module):
    def forward(self, x):
        result = torch.full((1, 6, 41), -12.)
        result[:, :, 40] = 0
        result[:, 2:4, 0] = 10
        return result


def runtime(monkeypatch, options):
    monkeypatch.setattr(wrapper, 'train_audio_transforms', Transform())
    return wrapper.align(torch.zeros(4608), ['word'], ['AA'],
                         np.array([[0, 1]]), np.array([[0, 1]]),
                         method=(Acoustic(), None, 'Baseline', False, 'cpu'),
                         verbose=False, return_details=True, alignment_options=options)


def test_constrained_decoder_honors_entering_window_and_retains_half_open_frames(monkeypatch):
    result = runtime(monkeypatch, {'mode': 'constrained', 'onset_windows': [[2, 3]],
                                  'word_windows': [[1, 5]], 'silence_intervals': [[0, .04]]})
    assert result[0] == [[2, 4]]
    details = result[2]
    assert details['constraints']['executed_mode'] == 'constrained'
    assert details['constraints']['satisfied'] is True
    assert details['fallback'] == []
    assert details['frame_interval_convention'] == 'half_open_phone_occupancy'
    assert details['constrained_dp']['phone_frames'] == [[2, 4]]
    assert details['constrained_dp']['visited_cells'] > 0
    assert details['physical_origin_calibrated'] is False


def test_unreachable_constraints_have_explicit_legacy_fallback_or_strict_failure(monkeypatch):
    from t2l.alignment.constrained_dp import NoAlignmentPathError
    monkeypatch.setattr(wrapper.utils, 'alignment', lambda *args: ([[1, 4]], -2.))
    options = {'mode': 'constrained', 'silence_intervals': [[0, 4608 / 22050]]}
    result = runtime(monkeypatch, options)
    assert result[0] == [[1, 4]]
    assert result[2]['constraints']['executed_mode'] == 'legacy'
    assert result[2]['constraints']['satisfied'] is False
    assert 'without satisfying' in result[2]['fallback'][0]
    with pytest.raises(NoAlignmentPathError):
        runtime(monkeypatch, dict(options, fallback_to_legacy=False))


@pytest.mark.parametrize('options', [
    {'word_windows': [[5, 1]]}, {'silence_intervals': [[0, float('nan')]]},
    {'silence_intervals': [[-.1, .1]]}, {'silence_intervals': [[0, 10]]},
    {'mode': 'legacy', 'word_windows': [[0, 6]]}, {'unknown': 1},
    {'fallback_to_legacy': 'yes'},
])
def test_invalid_constraints_are_not_silently_fallback(monkeypatch, options):
    def forbidden(*args):
        raise AssertionError('invalid configuration fell back to legacy')
    monkeypatch.setattr(wrapper.utils, 'alignment', forbidden)
    with pytest.raises(ValueError):
        runtime(monkeypatch, options)


@pytest.mark.parametrize('vocalize', [False, True])
def test_process_opt_in_metadata_decodes_once_and_uses_real_original_grid(monkeypatch, vocalize):
    calls, captured = [], {}
    def decode(path, sr=None):
        calls.append((path, sr))
        return torch.zeros(2, 48000), 48000
    def phones(text, language=None):
        assert language == 'en'
        return [('Hello', 'hello')]
    def align(*args, **kwargs):
        captured.update(kwargs)
        assert args[0].shape[-1] == 22050
        return [[1, 2]], None, {'input_samples': 22050}
    monkeypatch.setattr(pipeline, 'phonetize', phones)
    monkeypatch.setattr(pipeline, 'preprocess_audio', decode)
    monkeypatch.setattr(pipeline, '__vocalize',
                        lambda x, rate, target, *args: (torch.zeros(1, 22050), 22050))
    monkeypatch.setattr(pipeline.mtl_utils, 'gen_phone_gt_opt',
                        lambda p: (['AA'], None, np.array([[0, 1]]), np.array([[0, 1]])))
    monkeypatch.setattr(pipeline, 'align', align)
    result = pipeline.process(['Hello'], 'original.wav', vocalize=vocalize,
                              language='en', include_audio_metadata=True,
                              alignment_options={'mode': 'constrained'},
                              return_details=True, verbose=False)
    assert calls == [('original.wav', None)]
    assert result['original_audio']['sample_rate'] == 48000
    assert result['original_audio']['num_samples'] == 48000
    assert result['original_audio']['channels'] == 2
    assert result['units'][0]['text'] == 'Hello'
    assert captured['alignment_options'] == {'mode': 'constrained'}


def test_invalid_explicit_language_fails_before_audio_loading(monkeypatch):
    monkeypatch.setattr(pipeline, 'preprocess_audio', lambda *args: pytest.fail('audio loaded'))
    with pytest.raises(ValueError, match='language'):
        pipeline.process(['hello'], 'unused.wav', language='not-a-language')


def test_explicit_english_preserves_text_and_normalizes_pronunciation_without_language_model(monkeypatch):
    # Import a private module instance with the heavy language model stubbed.
    fast = SimpleNamespace(load_model=lambda _: SimpleNamespace(predict=lambda *args, **kwargs:
                            pytest.fail('explicit language invoked fastText')))
    monkeypatch.setitem(sys.modules, 'fasttext', fast)
    def unused_language(*args, **kwargs):
        pytest.fail('English pronunciation invoked a non-English language dependency')
    monkeypatch.setitem(sys.modules, 'pypinyin', SimpleNamespace(lazy_pinyin=unused_language))
    monkeypatch.setitem(sys.modules, 'pykakasi',
                        SimpleNamespace(kakasi=lambda: SimpleNamespace(convert=unused_language)))
    monkeypatch.setitem(sys.modules, 'kroman', SimpleNamespace(parse=unused_language))
    monkeypatch.setitem(sys.modules, 'cyrtranslit', SimpleNamespace(to_latin=unused_language))
    spec = importlib.util.spec_from_file_location('fixture_explicit_phonetic',
                                                 Path(pipeline.__file__).with_name('phonetic.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.phonetize('Hello WORLD', language='en') == [('Hello ', 'hello'), ('WORLD', 'world')]
    assert module.phonetize('Hello WORLD') == [('Hello WORLD', 'Hello WORLD')]
    assert module.phonetize('') == [('', '')]
    with pytest.raises(ValueError, match='language'):
        module.phonetize('hello', language='unsupported')
