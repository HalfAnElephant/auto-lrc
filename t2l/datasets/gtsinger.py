"""Read one published GTSinger segment without loading an acoustic model.

TextGrid word/phone intervals supply annotation candidates. JSON and MusicXML
remain separate sources: disagreements are reported, never silently repaired.
All boundaries require human review, regardless of source agreement.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
import wave
import xml.etree.ElementTree as ET
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional

from praatio import textgrid

from .classification import annotate_unit, audit_score
from .schema import AudioSegment, Boundary, LyricUnit, Phoneme


_TECHNIQUES = ("mix", "falsetto", "breathy", "breathe", "pharyngeal", "vibrato", "glissando")


def _error(path: Any, message: str) -> ValueError:
    return ValueError(f"{path}: {message}")


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, "expected a finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise _error(path, "expected a finite number") from exc
    if not math.isfinite(value):
        raise _error(path, "expected a finite number")
    return value


def _interval(start: Any, end: Any, duration: float, path: str) -> tuple:
    start, end = _number(start, path + ".start"), _number(end, path + ".end")
    if not 0 <= start < end <= duration:
        raise _error(path, f"interval [{start}, {end}] must satisfy 0 <= start < end <= {duration}")
    return start, end


def _sample(seconds: float, sample_rate: int) -> int:
    return int((Decimal(str(seconds)) * sample_rate).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _kind(label: str) -> str:
    if label.strip() in ("", "<SP>"):
        return "silence"
    if label.strip() == "<AP>":
        return "breath"
    return "lyric"


def _source(path: Path) -> Dict[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _files(root: Path, relative_stem: str) -> tuple:
    if not isinstance(relative_stem, str) or "\\" in relative_stem:
        raise _error(relative_stem, "expected a POSIX relative stem with six path components")
    parts = relative_stem.split("/")
    if len(parts) != 6 or any(part in ("", ".", "..") for part in parts):
        raise _error(relative_stem, "expected language/singer/technique/song/group/segment")
    root = root.resolve()
    stem = root.joinpath(*parts)

    def checked(suffix: str, required: bool = True) -> Path:
        path = Path(str(stem) + suffix).resolve()
        if not path.is_relative_to(root):
            raise _error(path, "source is outside the dataset root")
        if required and not path.is_file():
            raise _error(path, "required source file is missing")
        return path

    tg_candidates = [checked(suffix, False) for suffix in (".TextGrid", "_TextGrid")]
    existing = [path for path in tg_candidates if path.is_file()]
    if len(existing) != 1:
        raise _error(stem, "expected exactly one of .TextGrid and _TextGrid (missing or ambiguous)")
    return parts, {"wav": checked(".wav"), "textgrid": existing[0],
                   "json": checked(".json"), "musicxml": checked(".musicxml")}


def _audio_info(path: Path) -> tuple:
    try:
        with wave.open(str(path), "rb") as audio:
            sample_rate, num_samples = audio.getframerate(), audio.getnframes()
            frame_bytes = audio.getnchannels() * audio.getsampwidth()
            if audio.getcomptype() != "NONE" or sample_rate <= 0 or num_samples <= 0:
                raise _error(path, "expected non-empty uncompressed PCM audio")
            actual_bytes = 0
            while True:
                data = audio.readframes(65536)
                if not data:
                    break
                actual_bytes += len(data)
            if actual_bytes != num_samples * frame_bytes:
                raise _error(path, "truncated PCM data does not match the WAV frame count")
    except (wave.Error, EOFError, OSError) as exc:
        raise _error(path, f"cannot read PCM WAV: {exc}") from exc
    return sample_rate, num_samples


def _json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite JSON number {value}")
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value}")


def _read_json(path: Path, duration: float) -> List[Dict[str, Any]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"),
                          parse_float=_json_float, parse_constant=_reject_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise _error(path, f"cannot read JSON: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise _error(path, "expected a non-empty array of word objects")
    previous_end = 0.0
    for i, row in enumerate(rows):
        where = f"{path}[{i}]"
        if not isinstance(row, dict) or not isinstance(row.get("word"), str):
            raise _error(where, "expected a word object with a string word")
        start, end = _interval(row.get("start_time"), row.get("end_time"), duration, where)
        if start < previous_end:
            raise _error(where, "word intervals overlap or are out of order")
        previous_end = end
        ph = row.get("ph")
        if not isinstance(ph, list) or not ph or not all(isinstance(label, str) for label in ph):
            raise _error(where + ".ph", "expected a non-empty array of phoneme labels")
        for field in ("ph_start", "ph_end") + tuple(key for key in _TECHNIQUES if key in row):
            if not isinstance(row.get(field), list) or len(row[field]) != len(ph):
                raise _error(where + "." + field, "array length must match ph")
        prior = start
        for j, (ph_start, ph_end) in enumerate(zip(row["ph_start"], row["ph_end"])):
            a, b = _interval(ph_start, ph_end, duration, f"{where}.ph[{j}]")
            if a < prior or b > end:
                raise _error(f"{where}.ph[{j}]", "phoneme must be ordered and contained in its word")
            prior = b
        note_fields = ("note", "note_start", "note_end", "note_dur")
        if any(field in row for field in note_fields):
            notes = row.get("note")
            if not isinstance(notes, list):
                raise _error(where + ".note", "expected an array")
            for field in ("note_start", "note_end") + (("note_dur",) if "note_dur" in row else ()):
                if not isinstance(row.get(field), list) or len(row[field]) != len(notes):
                    raise _error(where + "." + field, "array length must match note")
            prior = start
            for j, note in enumerate(notes):
                _number(note, f"{where}.note[{j}]")
                a, b = _interval(row["note_start"][j], row["note_end"][j], duration, f"{where}.note[{j}]")
                if a < prior or b > end:
                    raise _error(f"{where}.note[{j}]", "recorded note interval must be ordered and contained in its word")
                prior = b
                if "note_dur" in row and _number(row["note_dur"][j], f"{where}.note_dur[{j}]") <= 0:
                    raise _error(f"{where}.note_dur[{j}]", "score duration must be positive")
    return rows


def _read_textgrid(path: Path, duration: float) -> Dict[str, List[Dict[str, Any]]]:
    try:
        grid = textgrid.openTextgrid(str(path), includeEmptyIntervals=True, reportingMode="error")
    except Exception as exc:
        raise _error(path, f"cannot parse TextGrid: {exc}") from exc
    _interval(grid.minTimestamp, grid.maxTimestamp, duration, str(path) + ":domain")
    tiers = {}
    for name in grid.tierNames:
        tier = grid.getTier(name)
        if not isinstance(tier, textgrid.IntervalTier):
            raise _error(f"{path}:{name}", "expected an IntervalTier")
        _interval(tier.minTimestamp, tier.maxTimestamp, duration, f"{path}:{name}:domain")
        entries, previous_end = [], 0.0
        for i, entry in enumerate(tier.entries):
            where = f"{path}:{name}[{i}]"
            start, end = _interval(entry.start, entry.end, duration, where)
            if start < previous_end:
                raise _error(where, "intervals overlap or are out of order")
            previous_end = end
            entries.append({"start_seconds": start, "end_seconds": end, "label": entry.label})
        tiers[name] = entries
    if not tiers.get("word") or not tiers.get("phone"):
        raise _error(path, "non-empty word and phone IntervalTiers are required")
    return tiers


def _read_score(path: Path) -> List[Dict[str, Any]]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise _error(path, f"cannot parse MusicXML: {exc}") from exc
    # Strip an optional MusicXML namespace on the parsed copy only.
    for element in root.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    if root.tag != "score-partwise":
        raise _error(path, "expected score-partwise MusicXML")
    has_performance_route = any(root.find(".//" + tag) is not None for tag in ("repeat", "ending", "segno", "coda")) or any(
        key in sound.attrib for sound in root.iter("sound")
        for key in ("dacapo", "dalsegno", "dalcoda", "tocoda", "fine")
    )
    notes = []
    for part in root.findall("part"):
        divisions = None
        for measure in part.findall("measure"):
            children = list(measure)
            # A terminal backup used to place a direction does not reorder notes.
            cursor_changed_before_note = any(
                child.tag in ("backup", "forward") and any(later.tag == "note" for later in children[index + 1:])
                for index, child in enumerate(children)
            )
            seen_note = False
            midmeasure_divisions = False
            for child in children:
                if child.tag == "attributes" and child.find("divisions") is not None and seen_note:
                    midmeasure_divisions = True
                if child.tag == "note":
                    seen_note = True
            value = measure.findtext("attributes/divisions")
            if value is not None:
                try:
                    divisions = _number(float(value), str(path) + ":divisions")
                except ValueError as exc:
                    raise _error(path, "invalid MusicXML divisions") from exc
                if divisions <= 0:
                    raise _error(path, "MusicXML divisions must be positive")
            for note in measure.findall("note"):
                duration = note.findtext("duration")
                if duration is not None:
                    try:
                        duration = _number(float(duration), str(path) + ":duration")
                    except ValueError as exc:
                        raise _error(path, "invalid MusicXML note duration") from exc
                    if duration <= 0:
                        raise _error(path, "MusicXML note duration must be positive")
                notes.append({
                    "part_id": part.get("id"), "measure": measure.get("number"),
                    "voice": note.findtext("voice"), "is_rest": note.find("rest") is not None,
                    "is_chord": note.find("chord") is not None, "is_grace": note.find("grace") is not None,
                    "has_note_after_time_cursor_change": cursor_changed_before_note,
                    "has_midmeasure_divisions": midmeasure_divisions,
                    "has_performance_route": has_performance_route,
                    "pitch": {key: note.findtext("pitch/" + key) for key in ("step", "alter", "octave")},
                    "duration_divisions": duration, "divisions_per_quarter": divisions,
                    "lyrics": [{"text": "".join(child.text or "" for child in lyric.findall("text")),
                                "syllabic": lyric.findtext("syllabic"), "number": lyric.get("number"),
                                "has_elision": lyric.find("elision") is not None}
                               for lyric in note.findall("lyric")],
                    "ties": [tie.get("type") for tie in note.findall("tie")],
                    "slurs": [slur.get("type") for slur in note.findall("notations/slur")],
                })
    if not notes:
        raise _error(path, "MusicXML contains no notes")
    return notes


def load_segment(root: Path, relative_stem: str, *, dataset_revision: Optional[str] = None,
                 song_id: Optional[str] = None) -> AudioSegment:
    """Read one language/singer/technique/song/group/stem quartet.

    The default song key groups title variants across singers, techniques and
    groups, but cannot resolve translated titles or aliases; callers can provide
    a canonical song_id. No split assignment or ground truth is inferred.
    """
    parts, paths = _files(Path(root), relative_stem)
    language, singer, technique, title, group, _ = parts
    if song_id is not None and (not isinstance(song_id, str) or not song_id.strip()):
        raise _error(relative_stem, "song_id must be a non-empty string")
    if dataset_revision is not None and not isinstance(dataset_revision, str):
        raise _error(relative_stem, "dataset_revision must be a string or None")
    if song_id is None:
        song_id = "/".join(" ".join(unicodedata.normalize("NFKC", value).casefold().split())
                           for value in (language, title))
    sample_rate, num_samples = _audio_info(paths["wav"])
    duration = num_samples / sample_rate
    rows = _read_json(paths["json"], duration)
    tiers = _read_textgrid(paths["textgrid"], duration)
    score_notes = _read_score(paths["musicxml"])
    issues = []

    def issue(code: str, path: str, message: str) -> None:
        issues.append({"code": code, "path": path, "message": message})

    if sample_rate != 48000:
        issue("unexpected_sample_rate", str(paths["wav"]), f"expected GTSinger 48000 Hz; retained actual {sample_rate} Hz")
    words, phones = tiers["word"], tiers["phone"]
    if len(rows) != len(words):
        issue("json_textgrid_word_count", str(paths["json"]), f"JSON has {len(rows)} words; TextGrid has {len(words)}")
    assigned = [[] for _ in words]
    word_index = 0
    for i, phone in enumerate(phones):
        while word_index < len(words) and phone["start_seconds"] >= words[word_index]["end_seconds"]:
            word_index += 1
        if (word_index == len(words) or phone["start_seconds"] < words[word_index]["start_seconds"]
                or phone["end_seconds"] > words[word_index]["end_seconds"]):
            raise _error(f"{paths['textgrid']}:phone[{i}]", "phoneme is not contained in exactly one word interval")
        assigned[word_index].append(phone)
    technique_labels = {}
    phone_intervals = [(x["start_seconds"], x["end_seconds"]) for x in phones]
    for name in _TECHNIQUES:
        if name in tiers:
            intervals = [(x["start_seconds"], x["end_seconds"]) for x in tiers[name]]
            if intervals != phone_intervals:
                issue("textgrid_technique_alignment", f"{paths['textgrid']}:{name}", "technique intervals do not match phone intervals")
            else:
                technique_labels[name] = {interval: entry["label"] for interval, entry in zip(intervals, tiers[name])}
    units, row_matches = [], []
    for i, word in enumerate(words):
        row = rows[i] if i < len(rows) else None
        unit_phones = [Phoneme(x["label"], x["start_seconds"], x["end_seconds"],
                               _sample(x["start_seconds"], sample_rate), _sample(x["end_seconds"], sample_rate))
                       for x in assigned[i]]
        expected = {"word": word["label"], "start_time": word["start_seconds"], "end_time": word["end_seconds"],
                        "ph": [x.label for x in unit_phones], "ph_start": [x.start_seconds for x in unit_phones],
                        "ph_end": [x.end_seconds for x in unit_phones]}
        row_aligned = row is not None and all(row[key] == value for key, value in expected.items())
        row_matches.append(row_aligned)
        if row is not None:
            for name, labels in technique_labels.items():
                if name in row:
                    expected[name] = [labels[(x.start_seconds, x.end_seconds)] for x in unit_phones]
            for key, value in expected.items():
                if row[key] != value:
                    issue("json_textgrid_mismatch", f"{paths['json']}[{i}].{key}",
                          "JSON differs from the corresponding TextGrid ordinal; both sources are preserved")
        kind = _kind(word["label"])
        sounding = next((phone for phone in unit_phones if _kind(phone.label) == "lyric"), None)
        boundary = None
        if kind == "lyric":
            if sounding is None:
                issue("lyric_without_phone", f"{paths['textgrid']}:word[{i}]", "no non-silence/breath phoneme supplies an onset candidate")
            else:
                boundary = Boundary(sounding.start_seconds, sounding.start_sample)
        unit = LyricUnit(word["label"], kind, word["start_seconds"], word["end_seconds"],
                               _sample(word["start_seconds"], sample_rate), _sample(word["end_seconds"], sample_rate),
                               unit_phones, boundary, row, i if row is not None else None,
                               first_sounding_phoneme=sounding if kind == "lyric" else None)
        flags = {}
        for name in ("breathy", "glissando"):
            flags[name] = {}
            if row is not None and name in row:
                flags[name]["json:" + name] = row[name]
            if name in technique_labels:
                flags[name]["textgrid:" + name] = [technique_labels[name][(phone.start_seconds, phone.end_seconds)]
                                                      for phone in unit_phones]
        annotate_unit(unit, language, flags=flags, row_aligned=row_aligned)
        units.append(unit)
    score_alignment = audit_score(units, score_notes, row_matches, issue, str(paths["json"]))
    return AudioSegment(relative_stem, song_id, language, singer, technique, group,
                        str(paths["wav"]), sample_rate, num_samples, units,
                        {name: _source(path) for name, path in paths.items()}, rows, tiers, score_notes,
                        dataset_revision=dataset_revision, validation_issues=issues,
                        score_alignment=score_alignment)
