"""Deterministic annotation screening, never an acoustic or human judgement."""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from .schema import LyricUnit


# CMUdict's published phone categories, with stop/liquid/semivowel names grouped.
# https://github.com/cmusphinx/cmudict/blob/master/cmudict.phones
_ARPABET = {
    **dict.fromkeys("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split(), "vowel"),
    **dict.fromkeys("B D G K P T".split(), "plosive"),
    **dict.fromkeys("DH F S SH TH V Z ZH".split(), "fricative"),
    **dict.fromkeys("CH JH".split(), "affricate"),
    **dict.fromkeys("M N NG".split(), "nasal"),
    **dict.fromkeys("L R W Y".split(), "approximant"),
    "HH": "aspirate",
}
_PINYIN_INITIALS = {
    **dict.fromkeys("b p d t g k".split(), "plosive"),
    **dict.fromkeys("f h s sh x".split(), "fricative"),
    **dict.fromkeys("z c zh ch j q".split(), "affricate"),
    **dict.fromkeys("m n".split(), "nasal"),
    **dict.fromkeys("l y w".split(), "approximant"),
}
_PINYIN_FINALS = set("a o e i u v ai ei ao ou an en ang eng ong er ia ie iao iu ian in iang ing iong ua uo uai ui uan un uang ueng ve van vn".split())
_SCREENING = "annotation_derived_screening; not acoustic fact or human ground truth"


def _marked(value: str, source: str, *, evidence: Optional[List[Any]] = None,
            rule: Optional[str] = None) -> Dict[str, Any]:
    return {"value": value, "source": source, "evidence": evidence or [],
            "rule": rule, "interpretation": _SCREENING}


def _phone_category(label: str, language: str) -> str:
    if language == "English":
        # Case is significant: lowercase unrecognized labels are not guessed.
        return _ARPABET.get(re.sub(r"[012]$", "", label), "unknown")
    if language == "Chinese":
        if label in _PINYIN_FINALS:
            return "vowel"
        return _PINYIN_INITIALS.get(label, "unknown")
    return "unknown"


def _han_character(text: str) -> bool:
    return len(text) == 1 and ("\u3400" <= text <= "\u4dbf" or "\u4e00" <= text <= "\u9fff"
                              or "\U00020000" <= text <= "\U000323af")


def _technique(name: str, unit: LyricUnit, flags: Dict[str, List[Any]],
               row_aligned: bool) -> Dict[str, Any]:
    if unit.kind != "lyric":
        return _marked("not_applicable", "textgrid:word_kind")
    evidence = []
    for source, values in flags.items():
        normalized = [str(value) for value in values]
        evidence.append({"source": source, "flags": normalized})
        if len(normalized) != len(unit.phonemes) or any(value not in ("0", "1") for value in normalized):
            return _marked("unknown", "unsupported_annotation_flags", evidence=evidence)
    if not evidence:
        return _marked("unknown", "annotation_flag_missing")
    if "json:" + name in flags and not row_aligned:
        return _marked("unknown", "json_textgrid_correspondence_unverified", evidence=evidence)
    if len({tuple(entry["flags"]) for entry in evidence}) != 1:
        return _marked("unknown", "annotation_sources_disagree", evidence=evidence)
    sounding = [i for i, phone in enumerate(unit.phonemes) if phone.label.strip() not in ("", "<SP>", "<AP>")]
    if not sounding:
        return _marked("unknown", "no_sounding_phoneme", evidence=evidence)
    values = evidence[0]["flags"]
    return _marked("present" if any(values[i] == "1" for i in sounding) else "absent",
                   "aligned_phone_annotations", evidence=evidence,
                   rule="any sounding phoneme flag is 1; all 0 means annotated absence")


def annotate_unit(unit: LyricUnit, language: str, *, flags: Dict[str, Dict[str, List[Any]]],
                  row_aligned: bool) -> None:
    """Add source-derived granularity and categories without changing candidates."""
    first = unit.first_sounding_phoneme
    if unit.kind != "lyric":
        unit_kind = "nonlexical"
    elif language == "Chinese" and _han_character(unit.text):
        unit_kind = "han_character"
    elif language == "English":
        unit_kind = "english_word"
    else:
        unit_kind = "source_unit"
    nuclei = [asdict(phone) for phone in unit.phonemes
              if _phone_category(phone.label, language) == "vowel"] if unit.kind == "lyric" else []
    unit.granularity = {
        "unit_kind": unit_kind, "source": "textgrid:word",
        "phoneme_nuclei": nuclei, "phoneme_nucleus_count": len(nuclei) if nuclei else None,
        "nucleus_source": "english_arpabet_vowels" if language == "English" else
                          "chinese_pinyin_final_labels" if language == "Chinese" else "unknown",
        "nucleus_scope": "annotated vowel/rime intervals only; not full syllable boundaries",
        "score_syllables": [], "phone_to_score_syllable": "unknown",
        "phone_to_score_syllable_reason": "lyric text and note pitches do not establish phone-to-score-syllable interval allocation",
    }
    unit.classifications = {
        "rest": _marked("present" if unit.kind == "silence" else "absent", "textgrid:word_kind"),
        "breath_event": _marked("present" if unit.kind == "breath" else "absent", "textgrid:word_kind"),
        "breathy": _technique("breathy", unit, flags.get("breathy", {}), row_aligned),
        "glissando": _technique("glissando", unit, flags.get("glissando", {}), row_aligned),
        "long_note": _marked("unknown" if unit.kind == "lyric" else "not_applicable", "score_mapping_unavailable"),
        "connected_singing": _marked("unknown" if unit.kind == "lyric" else "not_applicable", "not_observable_from_available_annotations"),
        "resinging": _marked("unknown" if unit.kind == "lyric" else "not_applicable", "not_observable_from_available_annotations"),
        "first_phoneme": _marked(
            _phone_category(first.label, language) if first is not None and unit.kind == "lyric" else
            "unknown" if unit.kind == "lyric" else "not_applicable",
            "textgrid:first_sounding_phoneme",
            evidence=[first.label] if first is not None and unit.kind == "lyric" else [],
            rule="CMUdict category or recognized Mandarin pinyin initial/final label; unsupported labels unknown"),
    }


def _pitch(note: Dict[str, Any]) -> Optional[int]:
    values = note["pitch"]
    if values.get("step") not in "C D E F G A B".split():
        return None
    try:
        octave = int(values["octave"])
        alter = float(values["alter"]) if values.get("alter") is not None else 0.0
    except (ValueError, TypeError):
        return None
    if not alter.is_integer() or not -1 <= octave <= 9:
        return None
    return 12 * (octave + 1) + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[values["step"]] + int(alter)


def _score_groups(notes: List[Dict[str, Any]]) -> tuple:
    if len({note["part_id"] for note in notes}) != 1 or len({note["voice"] for note in notes}) != 1:
        return [], "multiple_parts_or_voices"
    if any(note.get("has_performance_route") for note in notes):
        return [], "repeat_ending_or_performance_jump"
    if any(note.get("has_midmeasure_divisions") for note in notes):
        return [], "midmeasure_divisions_change"
    if any(note["is_chord"] or note["is_grace"] or note.get("has_note_after_time_cursor_change") for note in notes):
        return [], "chord_grace_or_time_cursor_change"
    groups, current = [], None
    tie_pitch, slur_active, syllabic_open = None, False, False
    for index, note in enumerate(notes):
        if note["is_rest"]:
            if tie_pitch is not None or slur_active or syllabic_open:
                return [], "unfinished_tie_slur_or_syllable_before_rest"
            current = None
            continue
        pitch = _pitch(note)
        if pitch is None:
            return [], "unsupported_or_missing_pitch"
        lyrics = note["lyrics"]
        if len(lyrics) > 1 or any(lyric.get("has_elision") for lyric in lyrics):
            return [], "multiple_lyric_lines_or_elision"
        ties, slurs = note["ties"], note["slurs"]
        if (len(ties) != len(set(ties)) or len(slurs) != len(set(slurs))
                or not set(ties) <= {"start", "stop"} or not set(slurs) <= {"start", "stop", "continue"}):
            return [], "unsupported_tie_or_slur_notation"
        has_text = bool(lyrics and lyrics[0]["text"])
        tied_from_previous = tie_pitch is not None and "stop" in ties and pitch == tie_pitch
        if (tie_pitch is not None) != ("stop" in ties) or (tie_pitch is not None and not tied_from_previous):
            return [], "tie_chain_pitch_or_start_stop_mismatch"
        if "stop" in slurs and not slur_active:
            return [], "slur_stop_without_start"
        if "continue" in slurs and not slur_active:
            return [], "slur_continue_without_start"
        if "start" in slurs and slur_active:
            return [], "nested_slur"
        if has_text:
            lyric = lyrics[0]
            syllabic = lyric["syllabic"] or "single"
            if syllabic not in ("single", "begin", "middle", "end"):
                return [], "unsupported_syllabic_value"
            if syllabic in ("middle", "end"):
                if not syllabic_open or current is None:
                    return [], "syllabic_continuation_without_begin"
                current["text"] += lyric["text"]
            else:
                if syllabic_open or tie_pitch is not None or slur_active:
                    return [], "new_lyric_inside_unfinished_chain"
                current = {"text": lyric["text"], "note_indices": [], "syllables": [],
                           "pitch_sequence": [], "duration_quarters": 0.0}
                groups.append(current)
            current["syllables"].append({"text": lyric["text"], "syllabic": syllabic,
                                          "source": "musicxml:lyric", "note_index": index})
            syllabic_open = syllabic in ("begin", "middle")
        elif current is None or not (tied_from_previous or slur_active):
            return [], "unlabelled_note_without_explicit_tie_or_slur"
        current["note_indices"].append(index)
        if not tied_from_previous:
            current["pitch_sequence"].append(pitch)
        duration, divisions = note["duration_divisions"], note["divisions_per_quarter"]
        if duration is None or divisions is None:
            current["duration_quarters"] = None
        elif current["duration_quarters"] is not None:
            current["duration_quarters"] += duration / divisions
            if not math.isfinite(current["duration_quarters"]):
                return [], "nonfinite_score_duration_quarters"
        tie_pitch = pitch if "start" in ties else None
        if "stop" in slurs:
            slur_active = False
        elif "start" in slurs:
            slur_active = True
    if tie_pitch is not None or slur_active or syllabic_open:
        return [], "unfinished_tie_slur_or_syllable"
    return groups, None


def audit_score(units: List[LyricUnit], notes: List[Dict[str, Any]],
                row_matches: List[bool], issue: Any, json_path: str) -> Dict[str, Any]:
    """Map complete explicit score lyric chains; retain unsupported cases as unmatched."""
    result = {"status": "unmatched", "reason": None, "mapped_units": 0,
              "pitch_audit": "unavailable", "time_axis": "score divisions/quarters only; no audio seconds",
              "mapping_rule": "complete ordered exact lyric match in one part/voice, explicit tie/slur continuations",
              "score_note_count": len(notes)}
    groups, reason = _score_groups(notes)
    lyric_indices = [i for i, unit in enumerate(units) if unit.kind == "lyric"]
    result["textgrid_lyric_texts"] = [units[i].text for i in lyric_indices]
    result["score_lyric_texts"] = [group["text"] for group in groups]
    if reason is None and [group["text"] for group in groups] != [units[i].text for i in lyric_indices]:
        reason = "complete_score_textgrid_lyric_sequence_mismatch"
    if reason is not None:
        result["reason"] = reason
        for unit in units:
            unit.score_mapping = {"status": "unmatched", "reason": reason} if unit.kind == "lyric" else {
                "status": "not_applicable", "reason": "no_audio_time_mapping_for_score_rests"}
        return result
    audits = []
    for i, group in zip(lyric_indices, groups):
        unit = units[i]
        row = unit.raw_annotations
        json_pitches = row.get("note") if row_matches[i] and row is not None else None
        audit = "unavailable" if json_pitches is None else "match" if json_pitches == group["pitch_sequence"] else "mismatch"
        audits.append(audit)
        unit.score_mapping = {"status": "matched", "source": "musicxml:explicit_lyric_chain",
                              **group, "json_pitch_sequence": json_pitches, "pitch_audit": audit}
        unit.granularity["score_syllables"] = group["syllables"]
        if audit == "mismatch":
            issue("json_musicxml_pitch_mismatch", f"{json_path}[{unit.source_json_index}].note",
                  "JSON note pitches differ from the explicitly matched MusicXML lyric chain after tie consolidation")
        quarters = group["duration_quarters"]
        if audit == "match" and quarters is not None:
            unit.classifications["long_note"] = _marked(
                "present" if quarters >= 2 else "absent", "musicxml:matched_lyric_chain",
                evidence=[{"duration_quarters": quarters, "note_indices": group["note_indices"]}],
                rule="matched lyric chain totals >=2 quarters; score screening rule, not a seconds/acoustic sustain definition")
        else:
            unit.classifications["long_note"] = _marked(
                "unknown", "score_pitch_audit_unverified" if audit != "match" else "score_duration_unavailable",
                evidence=[{"pitch_audit": audit, "duration_quarters": quarters}])
    for unit in units:
        if unit.kind != "lyric":
            unit.score_mapping = {"status": "not_applicable", "reason": "no_audio_time_mapping_for_score_rests"}
    result.update(status="matched", mapped_units=len(groups),
                  pitch_audit="mismatch" if "mismatch" in audits else "match" if audits and all(x == "match" for x in audits) else "unavailable")
    return result
