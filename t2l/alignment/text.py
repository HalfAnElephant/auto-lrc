"""Lossless lyric metadata, independent of inference and pronunciation libraries."""
from collections import Counter
import re
import unicodedata


LANGUAGE_ALIASES = {
    "zh": "zh", "zh-cn": "zh", "zh-tw": "zh", "chinese": "zh",
    "en": "en", "english": "en", "ja": "ja", "japanese": "ja",
    "ko": "ko", "korean": "ko", "ru": "ru", "russian": "ru",
}
WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)


def normalize_language(language):
    if language is None:
        return None
    if not isinstance(language, str):
        raise ValueError("language must be a supported language name, 'auto' or None")
    key = language.strip().lower()
    if key == "auto":
        return "auto"
    if key not in LANGUAGE_ALIASES:
        raise ValueError("Unsupported language: " + language)
    return LANGUAGE_ALIASES[key]


def lyric_lines(lyrics):
    if isinstance(lyrics, str):
        lines = lyrics.splitlines()
    elif isinstance(lyrics, (list, tuple)) and all(isinstance(line, str) for line in lyrics):
        lines = list(lyrics)
    else:
        raise ValueError("lyrics must be a string or a sequence of text lines")
    if not lines or not any(line.strip() for line in lines):
        raise ValueError("lyrics must contain non-empty text")
    return lines


def script_of(character):
    point = ord(character)
    if (0x3400 <= point <= 0x4DBF or 0x4E00 <= point <= 0x9FFF
            or 0xF900 <= point <= 0xFAFF or 0x20000 <= point <= 0x323AF):
        return "han"
    if 0x3040 <= point <= 0x30FF or 0x31F0 <= point <= 0x31FF or 0xFF66 <= point <= 0xFF9F:
        return "kana"
    if 0xAC00 <= point <= 0xD7AF or 0x1100 <= point <= 0x11FF or 0x3130 <= point <= 0x318F:
        return "hangul"
    if 0x0400 <= point <= 0x052F or 0x2DE0 <= point <= 0x2DFF or 0xA640 <= point <= 0xA69F:
        return "cyrillic"
    if character.isalpha() and "LATIN" in unicodedata.name(character, ""):
        return "latin"
    return "other" if character.isalpha() else "common"


def text_scripts(text):
    return sorted({script_of(character) for character in text} - {"common"})


def detect_language(text):
    """Script heuristic only; Latin script alone does not establish English."""
    scripts = set(text_scripts(text))
    if scripts and scripts <= {"han", "kana"} and "kana" in scripts:
        return "ja"
    return {frozenset({"han"}): "zh", frozenset({"hangul"}): "ko",
            frozenset({"cyrillic"}): "ru"}.get(frozenset(scripts), "unknown")


def tokenize_line(text, language=None):
    """Annotate spans without feeding replacement text back into the runtime."""
    selected = normalize_language(language)
    detected = detect_language(text)
    effective = detected if selected in (None, "auto") else selected
    if effective == "zh":
        spans = [(index, index + 1, "character") for index, character in enumerate(text)
                 if script_of(character) == "han"]
        # Retain non-Han lexical spans as unknown, without treating them as Chinese.
        for match in WORD.finditer(text):
            start = None
            for offset in range(match.start(), match.end() + 1):
                non_han = offset < match.end() and script_of(text[offset]) != "han"
                if non_han and start is None:
                    start = offset
                elif not non_han and start is not None:
                    spans.append((start, offset, "unknown"))
                    start = None
        spans.sort()
    elif effective == "en":
        spans = [(match.start(), match.end(), "word") for match in WORD.finditer(text)]
    else:
        # Japanese, Korean, Russian and unclassified Latin text keep whitespace
        # chunks: no language-specific word/character segmentation is asserted.
        spans = [(match.start(), match.end(), "unknown") for match in re.finditer(r"\S+", text)]
    return [{"text": text[start:end], "start_char": start, "end_char": end,
             "granularity": granularity} for start, end, granularity in spans]


def describe_lyrics(lyrics, language=None):
    lines = lyric_lines(lyrics)
    selected = normalize_language(language)
    counts, words = Counter(), Counter()
    result = []
    for index, line in enumerate(lines):
        counts[line] += 1
        tokens = tokenize_line(line, selected)
        for token in tokens:
            words[token["text"]] += 1
            token["occurrence"] = words[token["text"]]
        result.append({"input_line_index": index, "text": line, "occurrence": counts[line],
                       "scripts": text_scripts(line), "detected_language": detect_language(line),
                       "tokens": tokens})
    return {"original_lines": lines, "language_request": selected,
            "language_source": "explicit" if selected not in (None, "auto") else "script_heuristic",
            "occurrence_index_base": 1, "lines": result,
            "purpose": "Input metadata only; inference units remain authoritative"}


def describe_runtime_units(units, lyrics, language=None):
    """Identify actual returned units without substituting the input tokenization."""
    metadata = describe_lyrics(lyrics, language)
    grouped = {}
    for unit in units:
        grouped.setdefault(unit["line_index"], []).append(unit["text"])
    line_counts, line_info = Counter(), {}
    for line_index, texts in grouped.items():
        identity = tuple(texts)
        line_counts[identity] += 1
        # Only claim an input-line link when ordinal and literal text match.
        raw = metadata["original_lines"]
        input_index = line_index if (line_index < len(raw)
                                     and (raw[line_index] == "".join(texts)
                                          or raw[line_index] == " ".join(texts))) else None
        line_info[line_index] = {"line_occurrence": line_counts[identity],
                                 "input_line_index": input_index, "line_unit_texts": texts}
    word_counts, in_line, result = Counter(), Counter(), []
    for unit in units:
        text, line = unit["text"], unit["line_index"]
        word_counts[text] += 1
        selected = normalize_language(language)
        effective = detect_language(text) if selected in (None, "auto") else selected
        tokens = tokenize_line(text, selected)
        result.append({"unit_in_line": in_line[line], "word_occurrence": word_counts[text],
                       **line_info[line], "language": effective, "scripts": text_scripts(text),
                       "language_source": metadata["language_source"], "input_tokens": tokens,
                       "input_line_text": (metadata["original_lines"][line_info[line]["input_line_index"]]
                                           if line_info[line]["input_line_index"] is not None else None),
                       "unit_source": "process.return_details.units; not input-token substitution",
                       "occurrence_index_base": 1})
        in_line[line] += 1
    return result
