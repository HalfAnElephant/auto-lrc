"""Synthetic reader contracts; these do not establish annotation accuracy."""

import hashlib
import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest
from praatio import textgrid

from t2l.datasets import load_segment


STEM = "English/EN-Alto-1/Glissando/My  LOVE/Control_Group/0000"


def make_segment(tmp_path, *, stem=STEM, extension=".TextGrid", words=None, phones=None):
    path = tmp_path / stem
    path.parent.mkdir(parents=True)
    words = words or [(0, 0.1, "<SP>"), (0.1, 0.5, "street"),
                      (0.5, 0.6, "<AP>"), (0.6, 1, "安")]
    phones = phones or [(0, 0.1, "<SP>"), (0.1, 0.2, "S"), (0.2, 0.5, "T"),
                        (0.5, 0.6, "<AP>"), (0.6, 1, "an")]
    with wave.open(str(path) + ".wav", "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(3)
        audio.setframerate(48000)
        audio.writeframes(b"\x00" * 48000 * 3)
    grid = textgrid.Textgrid()
    grid.addTier(textgrid.IntervalTier("word", words, 0, 1))
    grid.addTier(textgrid.IntervalTier("phone", phones, 0, 1))
    grid.save(str(path) + extension, format="long_textgrid", includeBlankSpaces=True)
    rows = []
    for start, end, label in words:
        contained = [(a, b, text) for a, b, text in phones if a >= start and b <= end]
        rows.append({"word": label, "start_time": start, "end_time": end,
                     "ph": [x[2] for x in contained], "ph_start": [x[0] for x in contained],
                     "ph_end": [x[1] for x in contained], "breathy": ["0"] * len(contained),
                     "note": [60], "note_start": [start], "note_end": [end],
                     "note_dur": [2.0]})
    Path(str(path) + ".json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    Path(str(path) + ".musicxml").write_text('''<?xml version="1.0"?>
<score-partwise><part id="P1"><measure number="1">
<attributes><divisions>2</divisions></attributes>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration>
<lyric number="1"><syllabic>single</syllabic><text>street</text></lyric>
<notations><slur type="start"/></notations></note>
<note><rest/><duration>2</duration></note>
</measure></part></score-partwise>''', encoding="utf-8")
    return path, rows


def rewrite_json(path, rows):
    Path(str(path) + ".json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def test_source_fidelity_candidate_and_score_units(tmp_path):
    path, rows = make_segment(tmp_path)
    before = {suffix: Path(str(path) + suffix).read_bytes()
              for suffix in (".wav", ".json", ".TextGrid", ".musicxml")}
    result = load_segment(tmp_path, STEM, dataset_revision="fixed-revision").to_dict()
    json.dumps(result, allow_nan=False)
    assert result["song_id"] == "english/my love"
    assert result["dataset_revision"] == "fixed-revision"
    assert result["sample_rate"] == 48000 and result["num_samples"] == 48000
    assert result["source_recording_offset_samples"] is None
    assert result["review_status"] == "pending" and result["validation_issues"] == []
    assert result["raw_json"] == rows
    assert result["textgrid_tiers"]["word"][1]["start_seconds"] == 0.1
    silence, lyric, breath, chinese = result["units"]
    assert [x["kind"] for x in result["units"]] == ["silence", "lyric", "breath", "lyric"]
    assert silence["boundary"] is None and breath["boundary"] is None
    assert lyric["text"] == "street" and chinese["text"] == "安"
    assert lyric["start_sample"] == 4800 and lyric["end_sample"] == 24000
    assert lyric["boundary"] == {
        "candidate_seconds": 0.1, "candidate_sample": 4800, "source": "textgrid:first_phone",
        "human_truth_sample": None, "human_truth_seconds": None, "confidence": None,
        "review_status": "pending",
    }
    assert lyric["raw_annotations"]["note_dur"] == [2.0]
    assert result["score_notes"][0]["duration_divisions"] == 4
    assert result["score_notes"][0]["divisions_per_quarter"] == 2
    assert result["score_notes"][0]["lyrics"][0]["text"] == "street"
    assert result["score_notes"][1]["is_rest"] is True
    assert "seconds" not in result["score_notes"][0]
    for suffix, data in before.items():
        assert Path(str(path) + suffix).read_bytes() == data
    assert result["sources"]["textgrid"]["sha256"] == hashlib.sha256(before[".TextGrid"]).hexdigest()


def test_alternate_textgrid_and_canonical_song_id(tmp_path):
    make_segment(tmp_path, extension="_TextGrid")
    result = load_segment(tmp_path, STEM, song_id="canonical:westlife-my-love")
    assert result.song_id == "canonical:westlife-my-love"
    assert result.sources["textgrid"]["path"].endswith("0000_TextGrid")


def test_ambiguous_textgrid_is_rejected(tmp_path):
    path, _ = make_segment(tmp_path)
    Path(str(path) + "_TextGrid").write_bytes(Path(str(path) + ".TextGrid").read_bytes())
    with pytest.raises(ValueError, match="ambiguous"):
        load_segment(tmp_path, STEM)


def test_valid_json_disagreement_is_visible_not_repaired(tmp_path):
    path, rows = make_segment(tmp_path)
    rows[1]["ph"][0] = "SH"
    rows[1]["ph_start"][0] = 0.11
    rewrite_json(path, rows)
    result = load_segment(tmp_path, STEM).to_dict()
    assert len(result["validation_issues"]) == 2
    assert all(x["code"] == "json_textgrid_mismatch" for x in result["validation_issues"])
    assert result["units"][1]["phonemes"][0]["label"] == "S"
    assert result["units"][1]["boundary"]["candidate_sample"] == 4800
    assert result["raw_json"][1]["ph_start"][0] == 0.11
    assert result["units"][1]["boundary"]["human_truth_sample"] is None


@pytest.mark.parametrize("field,value,match", [
    ("ph_end", [0.2], "array length"),
    ("breathy", [], "array length"),
    ("ph_start", [0.09, 0.2], "contained"),
    ("ph_end", [0.3, 0.5], "ordered"),
    ("end_time", 1.1, "interval"),
    ("start_time", float("nan"), "nonfinite"),
    ("ph_start", [True, 0.2], "finite number"),
    ("note_end", [], "array length"),
])
def test_invalid_json_schema_fails_with_source_path(tmp_path, field, value, match):
    path, rows = make_segment(tmp_path)
    rows[1][field] = value
    rewrite_json(path, rows)
    with pytest.raises(ValueError, match=match) as error:
        load_segment(tmp_path, STEM)
    assert "0000.json" in str(error.value)


def test_textgrid_phone_crossing_word_boundary_is_rejected(tmp_path):
    path, _ = make_segment(tmp_path)
    grid = textgrid.Textgrid()
    grid.addTier(textgrid.IntervalTier("word", [(0, 0.5, "a"), (0.5, 1, "b")], 0, 1))
    grid.addTier(textgrid.IntervalTier("phone", [(0, 0.6, "A"), (0.6, 1, "B")], 0, 1))
    grid.save(str(path) + ".TextGrid", format="long_textgrid", includeBlankSpaces=True)
    with pytest.raises(ValueError, match="contained in exactly one word") as error:
        load_segment(tmp_path, STEM)
    assert "TextGrid:phone" in str(error.value)


def test_half_sample_rounding_and_first_sounding_phone(tmp_path):
    make_segment(tmp_path, words=[(0, 1, "a")],
                 phones=[(0, 0.10003125, "<SP>"), (0.10003125, 1, "AA")])
    result = load_segment(tmp_path, STEM)
    assert result.units[0].boundary.candidate_sample == 4802
    assert result.units[0].boundary.candidate_seconds == 0.10003125
    assert result.units[0].first_sounding_phoneme.label == "AA"
    assert result.units[0].first_sounding_phoneme.start_sample == 4802
    assert result.units[0].first_sounding_phoneme.end_sample == 48000
    assert result.units[0].first_sounding_phoneme.end_seconds == 1


def test_truncated_wav_is_not_accepted_as_declared_duration(tmp_path):
    path, _ = make_segment(tmp_path)
    wav = Path(str(path) + ".wav")
    wav.write_bytes(wav.read_bytes()[:-9])
    with pytest.raises(ValueError, match="truncated PCM"):
        load_segment(tmp_path, STEM)


def test_missing_source_and_path_escape_are_rejected(tmp_path):
    path, _ = make_segment(tmp_path)
    Path(str(path) + ".musicxml").unlink()
    with pytest.raises(ValueError, match="musicxml.*missing"):
        load_segment(tmp_path, STEM)
    with pytest.raises(ValueError, match="expected language"):
        load_segment(tmp_path, "../singer/tech/song/group/0000")


def test_word_count_disagreement_preserves_unpaired_source(tmp_path):
    path, rows = make_segment(tmp_path)
    rewrite_json(path, rows[:-1])
    result = load_segment(tmp_path, STEM)
    assert any(issue["code"] == "json_textgrid_word_count" for issue in result.validation_issues)
    assert len(result.units) == 4
    assert result.units[-1].raw_annotations is None


def test_technique_disagreement_preserves_both_flags(tmp_path):
    path, rows = make_segment(tmp_path)
    grid_path = str(path) + ".TextGrid"
    grid = textgrid.openTextgrid(grid_path, includeEmptyIntervals=True)
    grid.addTier(textgrid.IntervalTier("breathy", [(x.start, x.end, "1")
                                                  for x in grid.getTier("phone").entries], 0, 1))
    grid.save(grid_path, format="long_textgrid", includeBlankSpaces=True)
    result = load_segment(tmp_path, STEM)
    assert any(issue["path"].endswith("[1].breathy") for issue in result.validation_issues)
    assert result.units[1].raw_annotations["breathy"] == ["0", "0"]
    assert result.textgrid_tiers["breathy"][1]["label"] == "1"


def test_nonfinite_unknown_json_value_is_rejected(tmp_path):
    path, _ = make_segment(tmp_path)
    source = Path(str(path) + ".json")
    source.write_text(source.read_text(encoding="utf-8").replace('"word":', '"extra": 1e400, "word":', 1),
                      encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite JSON"):
        load_segment(tmp_path, STEM)


def test_invalid_musicxml_is_rejected_with_path(tmp_path):
    path, _ = make_segment(tmp_path)
    Path(str(path) + ".musicxml").write_text("<score-partwise>", encoding="utf-8")
    with pytest.raises(ValueError, match="musicxml.*cannot parse"):
        load_segment(tmp_path, STEM)


def test_import_does_not_load_inference_modules():
    result = subprocess.run([sys.executable, "-c", "import sys; import t2l.datasets; "
                             "assert 'torch' not in sys.modules; assert 't2l.phonetic' not in sys.modules"],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def score_xml(path, notes, *, prefix="", suffix=""):
    Path(str(path) + ".musicxml").write_text(
        '<score-partwise><part id="P1"><measure number="1">'
        '<attributes><divisions>2</divisions></attributes>' + prefix + notes + suffix +
        '</measure></part></score-partwise>', encoding="utf-8")


def score_note(text=None, *, step="C", alter=None, duration=2, syllabic="single", extra=""):
    lyric = '' if text is None else f'<lyric><syllabic>{syllabic}</syllabic><text>{text}</text></lyric>'
    accidental = '' if alter is None else f'<alter>{alter}</alter>'
    return (f'<note><pitch><step>{step}</step>{accidental}<octave>4</octave></pitch>'
            f'<duration>{duration}</duration>{lyric}{extra}</note>')


def test_additive_schema_keeps_old_constructor_defaults():
    from t2l.datasets.schema import LyricUnit

    unit = LyricUnit("a", "lyric", 0, 1, 0, 48000, [], None, None, None)
    assert unit.review_status == "pending" and unit.first_sounding_phoneme is None
    assert unit.granularity == {} and unit.classifications == {} and unit.score_mapping == {}


def test_nonlexical_and_unobserved_categories_do_not_imply_human_review(tmp_path):
    make_segment(tmp_path)
    segment = load_segment(tmp_path, STEM)
    silence, street, breath, _ = segment.units
    assert silence.classifications["rest"]["value"] == "present"
    assert breath.classifications["breath_event"]["value"] == "present"
    assert breath.classifications["breathy"]["value"] == "not_applicable"
    assert silence.first_sounding_phoneme is None and breath.first_sounding_phoneme is None
    assert street.classifications["breathy"]["value"] == "absent"
    assert street.classifications["glissando"]["value"] == "unknown"
    assert street.classifications["first_phoneme"]["value"] == "fricative"
    assert all(unit.classifications[key]["value"] == "unknown" for unit in segment.units if unit.kind == "lyric"
               for key in ("connected_singing", "resinging"))
    assert all("not acoustic fact" in item["interpretation"] for unit in segment.units
               for item in unit.classifications.values())
    assert all(unit.review_status == "pending" for unit in segment.units)


@pytest.mark.parametrize("label,category", [("AA1", "vowel"), ("B", "plosive"), ("CH", "affricate"),
                                           ("M", "nasal"), ("R", "approximant"), ("HH", "aspirate"),
                                           ("aa", "unknown"), ("UNRECOGNIZED", "unknown")])
def test_english_first_phone_category_is_inventory_based(tmp_path, label, category):
    make_segment(tmp_path, words=[(0, 1, "a")], phones=[(0, 1, label)])
    unit = load_segment(tmp_path, STEM).units[0]
    assert unit.classifications["first_phoneme"]["value"] == category


def test_english_word_and_vowel_nuclei_are_not_character_or_syllable_boundaries(tmp_path):
    path, _ = make_segment(tmp_path, words=[(0, 1, "empty")],
                           phones=[(0, 0.2, "EH1"), (0.2, 0.5, "M"), (0.5, 0.7, "T"), (0.7, 1, "IY0")])
    score_xml(path, score_note("empty"))
    unit = load_segment(tmp_path, STEM).units[0]
    assert unit.granularity["unit_kind"] == "english_word"
    assert unit.granularity["phoneme_nucleus_count"] == 2
    assert [phone["label"] for phone in unit.granularity["phoneme_nuclei"]] == ["EH1", "IY0"]
    assert unit.granularity["score_syllables"][0]["text"] == "empty"
    assert unit.granularity["score_syllables"][0]["syllabic"] == "single"
    assert unit.granularity["phone_to_score_syllable"] == "unknown"


def test_chinese_char_and_pinyin_final_do_not_split_multi_character_source_unit(tmp_path):
    stem = "Chinese/ZH-Alto-1/Breathy/title/Control_Group/0000"
    make_segment(tmp_path, stem=stem, words=[(0, 0.5, "离"), (0.5, 1, "再见")],
                 phones=[(0, 0.2, "l"), (0.2, 0.5, "i"), (0.5, 0.7, "z"), (0.7, 1, "ai")])
    first, second = load_segment(tmp_path, stem).units
    assert first.granularity["unit_kind"] == "han_character"
    assert first.granularity["phoneme_nucleus_count"] == 1
    assert first.classifications["first_phoneme"]["value"] == "approximant"
    assert second.granularity["unit_kind"] == "source_unit"
    assert second.text == "再见" and second.granularity["phone_to_score_syllable"] == "unknown"
    assert second.classifications["first_phoneme"]["value"] == "affricate"


def test_technique_flags_preserve_conflict_and_do_not_use_directory_name(tmp_path):
    path, rows = make_segment(tmp_path)
    rows[1]["glissando"] = ["0", "1"]
    rewrite_json(path, rows)
    grid_path = str(path) + ".TextGrid"
    grid = textgrid.openTextgrid(grid_path, includeEmptyIntervals=True)
    grid.addTier(textgrid.IntervalTier("breathy", [(x.start, x.end, "1")
                                                  for x in grid.getTier("phone").entries], 0, 1))
    grid.save(grid_path, format="long_textgrid", includeBlankSpaces=True)
    unit = load_segment(tmp_path, STEM).units[1]
    assert unit.classifications["breathy"]["value"] == "unknown"
    assert unit.classifications["breathy"]["source"] == "annotation_sources_disagree"
    assert len(unit.classifications["breathy"]["evidence"]) == 2
    assert unit.classifications["glissando"]["value"] == "present"


@pytest.mark.parametrize("unaligned", [True, False])
def test_unverified_or_unsupported_json_technique_flags_stay_unknown(tmp_path, unaligned):
    path, rows = make_segment(tmp_path)
    if unaligned:
        rows[1]["ph"][0] = "SH"
    else:
        rows[1]["breathy"] = ["maybe", "0"]
    rewrite_json(path, rows)
    unit = load_segment(tmp_path, STEM).units[1]
    assert unit.classifications["breathy"]["value"] == "unknown"


def test_explicit_melisma_and_ties_match_pitches_without_constructing_score_seconds(tmp_path):
    path, rows = make_segment(tmp_path, words=[(0, 1, "a")], phones=[(0, 1, "AA")])
    rows[0]["note"] = [60, 62]
    rows[0]["note_start"] = [0, 0.5]
    rows[0]["note_end"] = [0.5, 1]
    rows[0]["note_dur"] = [1, 1]
    rewrite_json(path, rows)
    score_xml(path, score_note("a", extra='<notations><slur type="start"/></notations>') +
              score_note(step="D", extra='<notations><slur type="stop"/></notations><tie type="start"/>') +
              score_note(step="D", extra='<tie type="stop"/>'))
    segment = load_segment(tmp_path, STEM).to_dict()
    mapping = segment["units"][0]["score_mapping"]
    assert segment["score_alignment"]["status"] == "matched"
    assert mapping["note_indices"] == [0, 1, 2]
    assert mapping["pitch_sequence"] == [60, 62] and mapping["pitch_audit"] == "match"
    assert mapping["duration_quarters"] == 3
    long_note = segment["units"][0]["classifications"]["long_note"]
    assert long_note["value"] == "present" and "score screening rule" in long_note["rule"]
    assert not any("seconds" in key for key in mapping)
    assert segment["source_recording_offset_samples"] is None
    assert segment["units"][0]["boundary"]["human_truth_sample"] is None


def test_score_syllabic_begin_end_is_preserved_without_phone_assignment(tmp_path):
    path, rows = make_segment(tmp_path, words=[(0, 1, "empty")], phones=[(0, 0.5, "EH1"), (0.5, 1, "IY0")])
    rows[0]["note"] = [60, 60]
    rows[0]["note_start"] = [0, 0.5]
    rows[0]["note_end"] = [0.5, 1]
    rows[0]["note_dur"] = [1, 1]
    rewrite_json(path, rows)
    score_xml(path, score_note("em", syllabic="begin") + score_note("pty", syllabic="end"))
    unit = load_segment(tmp_path, STEM).units[0]
    assert unit.score_mapping["status"] == "matched"
    assert [syllable["text"] for syllable in unit.granularity["score_syllables"]] == ["em", "pty"]
    assert unit.granularity["phone_to_score_syllable"] == "unknown"


def test_musicxml_pitch_disagreement_is_audited_without_repair(tmp_path):
    path, _ = make_segment(tmp_path, words=[(0, 1, "a")], phones=[(0, 1, "AA")])
    score_xml(path, score_note("a", step="D", duration=4))
    segment = load_segment(tmp_path, STEM)
    unit = segment.units[0]
    assert segment.score_alignment["status"] == "matched"
    assert segment.score_alignment["pitch_audit"] == "mismatch"
    assert unit.score_mapping["pitch_sequence"] == [62]
    assert unit.raw_annotations["note"] == [60]
    assert segment.validation_issues[0]["code"] == "json_musicxml_pitch_mismatch"
    assert unit.classifications["long_note"]["value"] == "unknown"


@pytest.mark.parametrize("notes,reason", [
    (score_note("a") + score_note(), "unlabelled_note_without_explicit_tie_or_slur"),
    (score_note("other"), "complete_score_textgrid_lyric_sequence_mismatch"),
    (score_note("a", extra='<notations><slur type="start"/></notations>'), "unfinished_tie_slur_or_syllable"),
    (score_note("a", extra='<tie type="start"/>') + score_note(step="D", extra='<tie type="stop"/>'),
     "tie_chain_pitch_or_start_stop_mismatch"),
    (score_note("a", extra='<notations><slur type="continue"/></notations>'), "slur_continue_without_start"),
    (score_note("a", extra='<chord/>'), "chord_grace_or_time_cursor_change"),
    (score_note("a", extra='<grace/>'), "chord_grace_or_time_cursor_change"),
    (score_note("a") + '<backup><duration>2</duration></backup>' + score_note(), "chord_grace_or_time_cursor_change"),
    (score_note("a") + '<attributes><divisions>4</divisions></attributes>' + score_note(), "midmeasure_divisions_change"),
    (score_note("a") + '<barline><repeat direction="backward"/></barline>', "repeat_ending_or_performance_jump"),
    (score_note("a", extra='<voice>1</voice>') + score_note(extra='<voice>2</voice>'), "multiple_parts_or_voices"),
])
def test_unsupported_score_mapping_retains_reason_and_unknown_categories(tmp_path, notes, reason):
    path, _ = make_segment(tmp_path, words=[(0, 1, "a")], phones=[(0, 1, "AA")])
    score_xml(path, notes)
    segment = load_segment(tmp_path, STEM)
    assert segment.score_alignment["status"] == "unmatched"
    assert segment.score_alignment["reason"] == reason
    assert segment.units[0].score_mapping == {"status": "unmatched", "reason": reason}
    assert segment.units[0].classifications["long_note"]["value"] == "unknown"
    assert segment.units[0].granularity["score_syllables"] == []
    assert segment.units[0].boundary.human_truth_seconds is None


def test_terminal_backup_for_tempo_direction_does_not_claim_score_audio_timing(tmp_path):
    path, _ = make_segment(tmp_path, words=[(0, 1, "a")], phones=[(0, 1, "AA")])
    score_xml(path, score_note("a"), suffix='<backup><duration>2</duration></backup>'
              '<direction><direction-type><metronome><per-minute>120</per-minute></metronome></direction-type></direction>')
    result = load_segment(tmp_path, STEM)
    assert result.score_alignment["status"] == "matched"
    assert result.units[0].classifications["long_note"]["value"] == "absent"
    assert result.score_notes[0]["has_note_after_time_cursor_change"] is False


def test_nonfinite_derived_score_duration_is_unmatched_not_serialized(tmp_path):
    path, _ = make_segment(tmp_path, words=[(0, 1, "a")], phones=[(0, 1, "AA")])
    score_xml(path, score_note("a", duration="1e308"))
    xml = Path(str(path) + ".musicxml")
    xml.write_text(xml.read_text().replace('<divisions>2</divisions>', '<divisions>1e-308</divisions>'))
    segment = load_segment(tmp_path, STEM)
    assert segment.score_alignment["reason"] == "nonfinite_score_duration_quarters"
    json.dumps(segment.to_dict(), allow_nan=False)


def test_json_ordinal_mismatch_prevents_score_pitch_and_long_note_claim(tmp_path):
    path, rows = make_segment(tmp_path, words=[(0, 1, "a")], phones=[(0, 1, "AA")])
    score_xml(path, score_note("a", duration=4))
    rows[0]["word"] = "other"
    rewrite_json(path, rows)
    segment = load_segment(tmp_path, STEM)
    assert segment.score_alignment["status"] == "matched"
    assert segment.score_alignment["pitch_audit"] == "unavailable"
    assert segment.units[0].score_mapping["json_pitch_sequence"] is None
    assert segment.units[0].classifications["long_note"]["value"] == "unknown"
