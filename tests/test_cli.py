import json
import os
import sys

import pytest

from main import build_parser, cli, to_standard_lrc


def test_to_standard_lrc_removes_word_timestamps_with_long_minutes():
    enhanced = "[100:01.234]<100:01.234>Hello<09:02.003> world"

    assert to_standard_lrc(enhanced) == "[100:01.234]Hello world"


def test_parser_defaults_and_integer_index():
    args = build_parser().parse_args(["lyrics.txt", "song.mp3", "-i", "2"])

    assert args.format == "lrc"
    assert args.line_only == 0
    assert args.vocalize == 1
    assert args.idx == 2
    assert args.aligner == 'mtl'
    assert args.json_output is None
    assert args.enhanced_lrc_output is None


@pytest.mark.parametrize(
    "arguments",
    [
        ["lyrics.txt", "song.mp3", "--format", "srt"],
        ["lyrics.txt", "song.mp3", "--line_only", "2"],
        ["lyrics.txt", "song.mp3", "--vocalize", "-1"],
        ["lyrics.txt", "song.mp3", "--idx", "4"],
        ["lyrics.txt", "song.mp3", "--model", "unknown"],
        ["lyrics.txt", "song.mp3", "--aligner", "hybrid"],
    ],
)
def test_parser_rejects_unsupported_values(arguments):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(arguments)

    assert exc_info.value.code == 2


@pytest.mark.parametrize('diagnostics', [False, True])
def test_cli_compatible_output_and_optional_diagnostics(monkeypatch, tmp_path, capsys, diagnostics):
    import ext.t2lutils as t2lutils
    from t2l import t2l

    lyrics = tmp_path / '歌词.txt'
    lyrics.write_text('你好\n', encoding='utf-8')
    audio = tmp_path / 'song.mp3'
    out_dir = tmp_path / 'output'
    enhanced = '[00:00.104]<00:00.104>你好'
    result = {'lrc': enhanced, 'units': [{'text': '你好', 'start_seconds': 3 * 256 / 22050 * 3}],
              'alignment': {'score': -2.0}}
    captured = {}

    def fake_process(lines, music_file, **kwargs):
        captured.update(kwargs)
        assert lines == ['你好\n']
        assert music_file == str(audio)
        return result if kwargs['return_details'] else enhanced

    monkeypatch.setattr(t2l, 'process', fake_process)
    monkeypatch.setattr(t2lutils, '__get_file_encoding', lambda path: 'utf-8')
    arguments = ['main.py', str(lyrics), str(audio), '-o', str(out_dir)]
    if diagnostics:
        arguments += ['--aligner', 'mtl_bdr', '--json-output', str(tmp_path / 'extra' / '对齐.json'),
                      '--enhanced-lrc-output', str(tmp_path / 'extra' / '逐字.lrc')]
    monkeypatch.setattr(sys, 'argv', arguments)

    cli()

    assert (out_dir / 'song.lrc').read_bytes() == '[00:00.104]你好'.encode('utf-8')
    assert capsys.readouterr().out == enhanced + '\n\n标准LRC已保存至: ' + str(out_dir / 'song.lrc') + '\n'
    assert captured['mtl_model'] == ('MTL_BDR' if diagnostics else 'MTL')
    assert captured['return_details'] is diagnostics
    if diagnostics:
        assert (tmp_path / 'extra' / '逐字.lrc').read_bytes() == enhanced.encode('utf-8')
        assert json.loads((tmp_path / 'extra' / '对齐.json').read_text(encoding='utf-8')) == result


@pytest.mark.parametrize('collision', ['lyrics', 'audio', 'standard', 'extra', 'symlink', 'hardlink'])
def test_cli_rejects_output_collisions_before_reading_inputs(monkeypatch, tmp_path, collision):
    lyrics = tmp_path / 'lyrics.txt'
    audio = tmp_path / 'song.mp3'
    lyrics.write_text('untouched', encoding='utf-8')
    audio.write_bytes(b'audio')
    out_dir = tmp_path / 'out'
    target = {
        'lyrics': lyrics, 'audio': audio, 'standard': out_dir / 'song.lrc',
        'extra': tmp_path / 'extra.lrc', 'symlink': tmp_path / 'alias',
        'hardlink': tmp_path / 'alias',
    }[collision]
    if collision == 'symlink':
        target.symlink_to(lyrics)
    if collision == 'hardlink':
        os.link(lyrics, target)
    arguments = ['main.py', str(lyrics), str(audio), '-o', str(out_dir),
                 '--json-output', str(target)]
    if collision == 'extra':
        arguments += ['--enhanced-lrc-output', str(target)]
    monkeypatch.setattr(sys, 'argv', arguments)

    with pytest.raises(SystemExit) as exc_info:
        cli()

    assert exc_info.value.code == 2
    assert lyrics.read_text(encoding='utf-8') == 'untouched'
    assert audio.read_bytes() == b'audio'
    assert not out_dir.exists()
