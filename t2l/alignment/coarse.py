"""Adapt existing MTL inference into explicitly uncalibrated boundary candidates."""
import copy
import json
import math
from pathlib import Path

from .candidates import BoundaryCandidate
from .text import describe_runtime_units, lyric_lines, normalize_language


METHODS = {"mtl": "MTL", "mtl_bdr": "MTL_BDR"}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return float(value)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


def _options(constraints):
    if constraints is None:
        return None
    if not isinstance(constraints, dict):
        raise ValueError("constraints must be a dictionary or None")
    if not constraints:
        return None
    options = copy.deepcopy(constraints)
    try:
        json.dumps(options, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("constraints must contain finite JSON values") from error
    options.setdefault("mode", "constrained")
    if options["mode"] not in ("legacy", "constrained"):
        raise ValueError("constraints mode must be legacy or constrained")
    if "fallback_to_legacy" in options and not isinstance(options["fallback_to_legacy"], bool):
        raise ValueError("fallback_to_legacy must be boolean")
    if options["mode"] == "legacy" and set(options) != {"mode"}:
        raise ValueError("Legacy mode cannot silently ignore constraints")
    return options


def _model(method, model):
    if not isinstance(method, str) or method.lower() not in METHODS:
        raise ValueError("method must be mtl or mtl_bdr")
    name = METHODS[method.lower()]
    if model is None:
        return name, name
    if not isinstance(model, tuple) or len(model) != 5:
        raise ValueError("model must be the five-item tuple returned by load_mtl_model")
    acoustic, boundary, kind, bdr, device = model
    if acoustic is None or kind != "MTL" or not isinstance(bdr, bool):
        raise ValueError("Preloaded model tuple is not a valid MTL model")
    if bdr != (name == "MTL_BDR") or (bdr and boundary is None):
        raise ValueError("Preloaded model and requested method disagree")
    return name, model


def _boundary(unit, name, sample_rate, num_samples, reasons):
    frame, seconds = unit.get(name + "_frame"), unit.get(name + "_seconds")
    if frame is not None:
        _integer(frame, name + "_frame")
    if seconds is not None:
        seconds = _number(seconds, name + "_seconds")
    if frame is None or seconds is None:
        reasons.append(name + "_alignment_missing")
    coordinate = None
    if seconds is not None:
        if 0 <= seconds <= num_samples / sample_rate:
            coordinate = round(seconds * sample_rate)
        elif name == "end" and seconds > num_samples / sample_rate:
            seconds = num_samples / sample_rate
            coordinate = num_samples
            reasons.append("end_clipped_to_original_audio_duration")
        else:
            reasons.append(name + "_outside_original_audio")
    return frame, seconds, coordinate


def candidates_from_details(details, lyrics, *, method, language=None, constraints=None,
                            search_radius_seconds=0.15):
    """Pure adapter; missing intervals survive as unaligned candidates."""
    radius = _number(search_radius_seconds, "search_radius_seconds")
    if radius <= 0:
        raise ValueError("search_radius_seconds must be positive")
    if not isinstance(details, dict) or not isinstance(details.get("units"), list) or not details["units"]:
        raise ValueError("Inference must return its complete non-empty unit list")
    original = details.get("original_audio")
    if not isinstance(original, dict):
        raise ValueError("Inference did not provide original_audio sample coordinates")
    sample_rate = _integer(original.get("sample_rate"), "original sample_rate", 1)
    num_samples = _integer(original.get("num_samples"), "original num_samples", 1)
    alignment = details.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError("Inference did not provide alignment metadata")
    reported_method = alignment.get("method")
    if reported_method is not None and reported_method != method:
        raise ValueError("Inference returned a different model method")
    score = alignment.get("score")
    if score is not None:
        score = _number(score, "global path score")
    score_kind = alignment.get("score_kind", "unknown_uncalibrated")
    if not isinstance(score_kind, str):
        raise ValueError("score_kind must be text")
    frame_resolution = alignment.get("frame_resolution_seconds")
    if frame_resolution is not None and _number(frame_resolution, "frame_resolution_seconds") <= 0:
        raise ValueError("frame_resolution_seconds must be positive")
    applied = alignment.get("constraints", {})
    if not isinstance(applied, dict):
        raise ValueError("Returned constraints must be an object")
    fallback = alignment.get("fallback", [])
    if not isinstance(fallback, list) or not all(isinstance(reason, str) for reason in fallback):
        raise ValueError("Returned fallback must be a list of explanations")
    decoder = applied.get("executed_mode", applied.get("mode", "legacy"))
    units = details["units"]
    last_line = -1
    for index, unit in enumerate(units):
        if not isinstance(unit, dict) or not isinstance(unit.get("text"), str):
            raise ValueError("Inference units must retain text")
        word = _integer(unit.get("word_index"), "word_index")
        line = _integer(unit.get("line_index"), "line_index")
        if word != index or line < last_line:
            raise ValueError("Inference unit identities must preserve sequential word and line order")
        last_line = line
        if not isinstance(unit.get("phonemes", []), (list, tuple)):
            raise ValueError("Unit phonemes must be a sequence")
    identities = describe_runtime_units(units, lyrics, language)
    candidates = []
    for unit, identity in zip(units, identities):
        reasons = []
        start_frame, start, start_sample = _boundary(unit, "start", sample_rate, num_samples, reasons)
        end_frame, end, end_sample = _boundary(unit, "end", sample_rate, num_samples, reasons)
        if ((start is not None and end is not None and end < start)
                or (start_frame is not None and end_frame is not None and end_frame < start_frame)):
            raise ValueError("Inference returned a reversed unit interval")
        search_window = None
        if start is not None and start_sample is not None and start_frame is not None:
            lower, upper = max(0.0, start - radius), min(num_samples / sample_rate, start + radius)
            search_window = {"start_seconds": lower, "end_seconds": upper,
                             "start_sample": max(0, math.floor(lower * sample_rate)),
                             "end_sample": min(num_samples, math.ceil(upper * sample_rate)),
                             "coordinate_kind": "original audio; proposed onset search only"}
        notes = list(fallback) + reasons
        unavailable = [reason for reason in reasons if reason != "end_clipped_to_original_audio_duration"]
        if score is None:
            notes.append("global_path_score_missing")
        candidates.append(BoundaryCandidate(
            candidate_id="{}:line:{}:word:{}".format(method.lower(), unit["line_index"], unit["word_index"]),
            word_index=unit["word_index"], line_index=unit["line_index"], unit_in_line=identity["unit_in_line"],
            word_occurrence=identity["word_occurrence"], line_occurrence=identity["line_occurrence"],
            text=unit["text"], language=identity["language"], source=method.lower(),
            start_seconds=start, end_seconds=end,
            raw_frames={"start": start_frame, "end": end_frame,
                        "start_seconds": unit.get("start_seconds"), "end_seconds": unit.get("end_seconds"),
                        "convention": alignment.get("frame_interval_convention", "legacy_as_returned"),
                        "frame_resolution_seconds": frame_resolution},
            score=score, score_kind=score_kind, original_sample_rate=sample_rate,
            original_num_samples=num_samples, start_sample=start_sample, end_sample=end_sample,
            phonemes=list(unit.get("phonemes", [])), search_window=search_window,
            confidence=None, confidence_status="alignment_unavailable" if unavailable else "uncalibrated",
            status="unaligned" if unavailable else "candidate", decoder=decoder,
            constraints={"requested": copy.deepcopy(constraints), "runtime": copy.deepcopy(applied)},
            fallback=notes, text_metadata=identity))
    return candidates


def coarse_align(audio, lyrics, *, method="mtl", language=None, vocalize=False,
                 model=None, constraints=None, search_radius_seconds=0.15, verbose=False):
    """Generate one model's coarse candidates without calibration or fusion."""
    if not isinstance(audio, (str, Path)):
        raise ValueError("audio must be a file path")
    audio_path = Path(audio)
    if not audio_path.is_file():
        raise ValueError("audio must identify an existing file")
    lines = lyric_lines(lyrics)
    selected_language = normalize_language(language)
    if not isinstance(vocalize, bool) or not isinstance(verbose, bool):
        raise ValueError("vocalize and verbose must be boolean")
    radius = _number(search_radius_seconds, "search_radius_seconds")
    if radius <= 0:
        raise ValueError("search_radius_seconds must be positive")
    name, loaded = _model(method, model)
    options = _options(constraints)
    # Import only at the inference boundary; metadata/candidate tests need no models.
    from t2l.t2l import process
    details = process(lines, str(audio_path), mtl_model=loaded, vocalize=vocalize,
                      verbose=verbose, return_details=True, language=selected_language,
                      alignment_options=options, include_audio_metadata=True)
    return candidates_from_details(details, lines, method=name, language=selected_language,
                                   constraints=options, search_radius_seconds=radius)
