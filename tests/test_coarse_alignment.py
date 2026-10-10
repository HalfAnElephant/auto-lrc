import copy
import json
import sys
import types

import pytest

from t2l.alignment import BoundaryCandidate, coarse_align, describe_lyrics
from t2l.alignment.coarse import candidates_from_details
from t2l.alignment.text import detect_language, tokenize_line


def details(texts=("love", "love", "love", "love")):
    return {
        "original_audio": {"sample_rate": 48000, "num_samples": 48000},
        "alignment": {"method": "MTL", "score": -12.5, "score_kind": "whole_path_log_score_uncalibrated",
                      "frame_resolution_seconds": 0.1, "frame_interval_convention": "legacy_next_word_or_terminal_frame"},
        "units": [{"text": text, "word_index": index, "line_index": index // 2,
                   "phonemes": [12, 18], "start_frame": index, "end_frame": index + 1,
                   "start_seconds": index * 0.1, "end_seconds": (index + 1) * 0.1}
                  for index, text in enumerate(texts)],
    }


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture; inference mocked")
    observed = []
    returned = details()
    module = types.ModuleType("t2l.t2l")
    def process(*args, **kwargs):
        observed.append((args, kwargs))
        return copy.deepcopy(returned)
    module.process = process
    monkeypatch.setitem(sys.modules, "t2l.t2l", module)
    return audio, returned, observed


def test_public_api_preserves_actual_units_and_legacy_arguments(runtime):
    audio, returned, observed = runtime
    candidates = coarse_align(audio, "love love\nlove love", language="English")
    assert len(candidates) == 4 and all(isinstance(row, BoundaryCandidate) for row in candidates)
    args, kwargs = observed[0]
    assert args == (["love love", "love love"], str(audio))
    assert kwargs == {"mtl_model": "MTL", "vocalize": False, "verbose": False,
                      "return_details": True, "language": "en", "alignment_options": None,
                      "include_audio_metadata": True}
    assert [row.text for row in candidates] == [unit["text"] for unit in returned["units"]]
    assert [row.word_occurrence for row in candidates] == [1, 2, 3, 4]
    assert [row.line_occurrence for row in candidates] == [1, 1, 2, 2]
    assert [row.unit_in_line for row in candidates] == [0, 1, 0, 1]
    assert all(row.confidence is None and row.confidence_status == "uncalibrated" for row in candidates)
    assert all(row.score == -12.5 and row.source == "mtl" for row in candidates)
    assert [row.start_sample for row in candidates] == [0, 4800, 9600, 14400]
    assert candidates[0].search_window == {"start_seconds": 0.0, "end_seconds": 0.15,
                                           "start_sample": 0, "end_sample": 7200,
                                           "coordinate_kind": "original audio; proposed onset search only"}
    assert candidates[-1].text_metadata["input_line_index"] == 1
    assert len({row.candidate_id for row in candidates}) == 4
    json.dumps([row.to_dict() for row in candidates], allow_nan=False)


def test_model_reuse_and_constraint_fallback_are_explicit(runtime):
    audio, returned, observed = runtime
    acoustic, boundary = object(), object()
    model = (acoustic, boundary, "MTL", True, "cpu")
    returned["alignment"].update(method="MTL_BDR", constraints={"mode": "constrained", "executed_mode": "legacy"},
                                 fallback=["No constrained path; used legacy decoder"])
    options = {"onset_windows": [[0, 2], [1, 3], [2, 4], [3, 5]], "fallback_to_legacy": True}
    original = copy.deepcopy(options)
    candidates = coarse_align(audio, ["love love", "love love"], method="mtl_bdr", model=model,
                              constraints=options, vocalize=True)
    assert observed[0][1]["mtl_model"] is model
    assert observed[0][1]["alignment_options"] == {"mode": "constrained", **options}
    assert options == original
    assert all(row.source == "mtl_bdr" and row.decoder == "legacy" for row in candidates)
    assert all(row.fallback == ["No constrained path; used legacy decoder"] for row in candidates)
    assert candidates[0].constraints["requested"]["mode"] == "constrained"
    assert candidates[0].constraints["runtime"]["executed_mode"] == "legacy"


@pytest.mark.parametrize("unit_update", [{"start_frame": None, "start_seconds": None},
                                         {"end_frame": None}, {"start_seconds": None}])
def test_missing_boundaries_survive_as_uncalibrated_unaligned_candidates(unit_update):
    result = details()
    result["units"][2].update(unit_update)
    candidates = candidates_from_details(result, ["love love", "love love"], method="MTL", language="en")
    assert len(candidates) == 4
    missing = candidates[2]
    assert missing.text == "love" and missing.word_index == 2
    assert missing.status == "unaligned" and missing.confidence is None
    assert missing.confidence_status == "alignment_unavailable"
    assert any("alignment_missing" in reason for reason in missing.fallback)
    if "start_frame" in unit_update or "start_seconds" in unit_update:
        assert missing.search_window is None


def test_missing_keys_are_retained_and_no_origin_is_invented():
    result = details()
    for key in ("start_frame", "end_frame", "start_seconds", "end_seconds"):
        result["units"][0].pop(key)
    first = candidates_from_details(result, ["love love", "love love"], method="MTL")[0]
    assert first.start_seconds is None and first.end_seconds is None
    assert first.start_sample is None and first.end_sample is None
    assert first.raw_frames["start"] is None and first.raw_frames["end"] is None
    assert first.search_window is None and first.status == "unaligned"


def test_end_is_clipped_with_raw_half_open_frame_and_seconds_preserved():
    result = details(("last",))
    result["units"][0].update(start_frame=9, end_frame=11, start_seconds=0.9, end_seconds=1.1)
    result["alignment"].update(frame_interval_convention="half_open_phone_occupancy",
                               constraints={"mode": "constrained", "executed_mode": "constrained"})
    candidate = candidates_from_details(result, "last", method="MTL", search_radius_seconds=0.2)[0]
    assert candidate.end_seconds == 1.0 and candidate.end_sample == 48000
    assert candidate.raw_frames["end"] == 11 and candidate.raw_frames["end_seconds"] == 1.1
    assert candidate.raw_frames["convention"] == "half_open_phone_occupancy"
    assert candidate.search_window["end_seconds"] == 1.0
    assert candidate.search_window["end_sample"] == 48000
    assert candidate.status == "candidate" and candidate.decoder == "constrained"
    assert candidate.fallback == ["end_clipped_to_original_audio_duration"]


def test_out_of_range_start_is_not_clipped_into_a_valid_onset():
    result = details(("word",))
    result["units"][0].update(start_seconds=-0.1)
    candidate = candidates_from_details(result, "word", method="MTL")[0]
    assert candidate.start_seconds == -0.1 and candidate.start_sample is None
    assert candidate.status == "unaligned" and candidate.search_window is None
    assert "start_outside_original_audio" in candidate.fallback


def test_metadata_does_not_replace_runtime_units_with_english_tokens():
    result = details(("can't!", "sing"))
    candidates = candidates_from_details(result, "can't! sing", method="MTL", language="en")
    assert [row.text for row in candidates] == ["can't!", "sing"]
    assert candidates[0].text_metadata["input_tokens"][0]["text"] == "can't"
    assert candidates[0].text_metadata["input_line_text"] == "can't! sing"
    punctuation = describe_lyrics("  can't! sing  ", "en")
    assert punctuation["original_lines"] == ["  can't! sing  "]
    assert punctuation["lines"][0]["tokens"][0]["start_char"] == 2
    assert punctuation["lines"][0]["tokens"][0]["text"] == "can't"


def test_input_line_mapping_is_unknown_after_runtime_line_normalization():
    result = details(("word",))
    candidate = candidates_from_details(result, ["[ar:metadata]", "word"], method="MTL", language="en")[0]
    assert candidate.text_metadata["input_line_index"] is None
    assert candidate.text_metadata["input_line_text"] is None
    assert candidate.text == "word"


def test_chinese_character_metadata_retains_original_text_and_occurrences():
    report = describe_lyrics(["你，你", "你，你"], "zh")
    assert report["original_lines"] == ["你，你", "你，你"]
    assert [line["occurrence"] for line in report["lines"]] == [1, 2]
    tokens = [token for line in report["lines"] for token in line["tokens"]]
    assert [token["text"] for token in tokens] == ["你"] * 4
    assert [token["occurrence"] for token in tokens] == [1, 2, 3, 4]
    assert all(token["granularity"] == "character" for token in tokens)
    assert report["lines"][0]["tokens"][1]["start_char"] == 2


@pytest.mark.parametrize("text,expected", [("日本の歌", "ja"), ("中文", "zh"), ("한국어", "ko"),
                                          ("песня", "ru"), ("English", "unknown"), ("你 love", "unknown")])
def test_script_heuristics_are_labeled_and_do_not_guess_latin_language(text, expected):
    assert detect_language(text) == expected
    report = describe_lyrics(text)
    assert report["language_source"] == "script_heuristic"


@pytest.mark.parametrize("text,language", [("日本の歌", "ja"), ("한국어", "ko"), ("песня", "ru")])
def test_other_scripts_are_not_force_split_into_chinese_characters(text, language):
    assert tokenize_line(text, language) == [{"text": text, "start_char": 0, "end_char": len(text),
                                             "granularity": "unknown"}]


@pytest.mark.parametrize("kwargs", [{"method": "baseline"}, {"language": "xx"}, {"search_radius_seconds": -1},
                                    {"search_radius_seconds": float("nan")}, {"vocalize": 1},
                                    {"constraints": {"mode": "legacy", "onset_windows": []}},
                                    {"constraints": {"fallback_to_legacy": 1}}, {"constraints": []},
                                    {"model": (object(), None, "MTL", True, "cpu")},
                                    {"model": (object(), None, "MTL", False, "cpu"), "method": "mtl_bdr"}])
def test_invalid_public_input_is_refused_before_inference(runtime, kwargs):
    audio, _, observed = runtime
    with pytest.raises(ValueError):
        coarse_align(audio, "words", **kwargs)
    assert observed == []


@pytest.mark.parametrize("lyrics", ["", " \n ", [], [1], {"words": "hello"}])
def test_invalid_lyrics_are_refused_before_inference(runtime, lyrics):
    audio, _, observed = runtime
    with pytest.raises(ValueError):
        coarse_align(audio, lyrics)
    assert observed == []


def test_unknown_original_grid_is_not_replaced_by_model_sample_rate():
    result = details()
    result.pop("original_audio")
    result["alignment"]["sample_rate"] = 22050
    with pytest.raises(ValueError, match="original_audio"):
        candidates_from_details(result, "words", method="MTL")


@pytest.mark.parametrize("alteration", ["nan_score", "reversed", "bad_index", "wrong_method"])
def test_invalid_runtime_metadata_is_not_presented_as_candidates(alteration):
    result = details()
    if alteration == "nan_score":
        result["alignment"]["score"] = float("nan")
    elif alteration == "reversed":
        result["units"][0]["start_seconds"] = 0.2
    elif alteration == "bad_index":
        result["units"][0]["word_index"] = 3
    else:
        result["alignment"]["method"] = "MTL_BDR"
    with pytest.raises(ValueError):
        candidates_from_details(result, "words", method="MTL")


def test_serialized_candidate_is_independent_of_mutable_input():
    result = details(("word",))
    options = {"mode": "constrained", "onset_windows": [[0, 2]]}
    candidate = candidates_from_details(result, "word", method="MTL", constraints=options)[0]
    result["units"][0]["phonemes"].append(99)
    options["onset_windows"][0][0] = 10
    snapshot = candidate.to_dict()
    snapshot["phonemes"].append(77)
    assert candidate.phonemes == [12, 18]
    assert candidate.constraints["requested"]["onset_windows"] == [[0, 2]]
