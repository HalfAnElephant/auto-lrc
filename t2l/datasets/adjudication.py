"""Bind explicit human boundary decisions to immutable blind review snapshots."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
import re

from .annotations import IDENTITY_FIELDS, LABEL_FIELDS, row_key, validate_label


_ROW_FIELDS = set(IDENTITY_FIELDS + LABEL_FIELDS)
_POLICY = {
    "agreement": "retain_first_review_with_second_confirmation",
    "disagreement": "explicit_human_adjudication_or_pending",
    "ambiguity": "exclude_without_forcing_a_single_sample",
    "incomplete_reviews": "remain_pending",
    "independence": "Reviewer identities do not establish independence or an isolated test set.",
    "timing": "Only human original-WAV sample positions are used; no source annotations or models.",
}


def _canonical_json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Review data must be finite JSON: " + str(error)) from None


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def _positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _agreement_limit(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            (isinstance(value, float) and not math.isfinite(value)) or value < 0):
        raise ValueError("agreement_limit_ms must be finite and nonnegative")
    return Decimal(value) if isinstance(value, int) else Decimal(str(value))


def _within_limit(first, second, limit):
    # Integer cross multiplication avoids rounding at a fractional-sample limit.
    numerator, denominator = limit.as_integer_ratio()
    return (abs(first["human_truth_sample"] - second["human_truth_sample"]) * 1000 * denominator
            <= numerator * first["sample_rate"])


def _label_evidence(row):
    return {field: deepcopy(row[field]) for field in LABEL_FIELDS}


def _pending(row):
    result = {field: row[field] for field in IDENTITY_FIELDS}
    result.update(dict.fromkeys(LABEL_FIELDS))
    result["review_status"] = "pending"
    return result


def _validate_session(document, name):
    if not isinstance(document, dict):
        raise ValueError(name + " must be a blind review document")
    if (type(document.get("schema_version")) is not int or document["schema_version"] != 1 or
            document.get("protocol") != "blind"):
        raise ValueError(name + " must use blind protocol schema_version 1")
    definition = document.get("onset_definition")
    if not isinstance(definition, str) or not definition.strip():
        raise ValueError(name + " needs an onset_definition")
    if not _sha256(document.get("form_fingerprint")):
        raise ValueError(name + " needs a 64-hex form_fingerprint")
    revision = document.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise ValueError(name + " has an invalid revision")
    if not isinstance(document.get("history"), list):
        raise ValueError(name + " history must be an array")
    sources = document.get("audio_sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError(name + " needs original audio_sources")
    for segment_id, audio in sources.items():
        if (not isinstance(segment_id, str) or not segment_id or not isinstance(audio, dict) or
                set(audio) != {"sha256", "sample_rate", "num_samples"}):
            raise ValueError(name + " has invalid original audio metadata")
        if not _sha256(audio["sha256"]):
            raise ValueError(name + ": " + segment_id + " needs a 64-hex original WAV SHA256")
        if not _positive_integer(audio["sample_rate"]) or not _positive_integer(audio["num_samples"]):
            raise ValueError(name + ": " + segment_id + " needs positive integer sample_rate/num_samples")
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(name + " needs a nonempty rows array")
    raw, validated = {}, {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != _ROW_FIELDS:
            raise ValueError(name + " rows must contain only standard identity and label fields")
        key = row_key(row)
        if key in raw:
            raise ValueError(name + " has duplicate boundary identity")
        for field in ("song_id", "language", "audio_path"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(name + " has invalid identity: " + field)
        if not isinstance(row["text"], str):
            raise ValueError(name + " has invalid identity: text")
        audio = sources.get(row["segment_id"])
        if (audio is None or not _positive_integer(row["sample_rate"]) or
                row["sample_rate"] != audio["sample_rate"]):
            raise ValueError(name + " row and original WAV timeline disagree")
        try:
            validated[key] = validate_label(row, row, audio["num_samples"])
        except ValueError as error:
            raise ValueError(name + ": " + str(key) + ": " + str(error)) from None
        raw[key] = row
    if set(sources) != {key[0] for key in raw}:
        raise ValueError(name + " original audio coverage must match the rows")
    digest = hashlib.sha256(_canonical_json(document)).hexdigest()
    return raw, validated, digest


def _paired_reviews(first_document, second_document, agreement_limit_ms):
    limit = _agreement_limit(agreement_limit_ms)
    first_raw, first, first_digest = _validate_session(first_document, "First review")
    second_raw, second, second_digest = _validate_session(second_document, "Second review")
    for field in ("onset_definition", "form_fingerprint", "audio_sources"):
        if _canonical_json(first_document[field]) != _canonical_json(second_document[field]):
            raise ValueError("Blind reviews disagree on " + field)
    if set(first) != set(second):
        raise ValueError("Blind review coverage must match")
    disputed = []
    for key, row in first.items():
        other = second[key]
        identity = {field: row[field] for field in IDENTITY_FIELDS}
        other_identity = {field: other[field] for field in IDENTITY_FIELDS}
        if _canonical_json(identity) != _canonical_json(other_identity):
            raise ValueError("Blind review identity changed: " + str(key))
        if row["review_status"] == "pending" or other["review_status"] == "pending":
            continue
        if row["reviewer"] == other["reviewer"]:
            raise ValueError("Completed review pairs require distinct actual reviewer identities: " + str(key))
        if (row["review_status"] == "ambiguous" or other["review_status"] == "ambiguous" or
                not _within_limit(row, other, limit)):
            disputed.append(key)
    metadata = {
        "schema_version": 1, "protocol": "boundary_adjudication",
        "onset_definition": first_document["onset_definition"],
        "form_fingerprint": first_document["form_fingerprint"],
        "audio_sources": deepcopy(first_document["audio_sources"]),
        "first_review_sha256": first_digest, "second_review_sha256": second_digest,
        "agreement_limit_ms": int(limit) if limit == limit.to_integral_value() else float(limit),
        "policy": deepcopy(_POLICY),
        "independent_adjudicated_test_set": False,
    }
    return metadata, first_raw, second_raw, first, second, disputed


def build_adjudication_form(first_document, second_document, *, agreement_limit_ms=5):
    """Return unlabelled decisions for complete, disputed or ambiguous pairs only."""
    metadata, first_raw, second_raw, first, _, disputed = _paired_reviews(
        first_document, second_document, agreement_limit_ms)
    decisions = []
    for key in disputed:
        decisions.append(dict(_pending(first[key]), first_review=_label_evidence(first_raw[key]),
                              second_review=_label_evidence(second_raw[key])))
    return dict(metadata, decisions=decisions)


def resolve_adjudication(first_document, second_document, decisions_document, *, agreement_limit_ms=5):
    """Export human references; unresolved disputes and incomplete reviews stay pending."""
    metadata, first_raw, second_raw, first, second, disputed = _paired_reviews(
        first_document, second_document, agreement_limit_ms)
    if not isinstance(decisions_document, dict) or set(decisions_document) != set(metadata) | {"decisions"}:
        raise ValueError("Adjudication document metadata fields changed")
    for field, expected in metadata.items():
        if _canonical_json(decisions_document[field]) != _canonical_json(expected):
            raise ValueError("Adjudication metadata or review snapshot changed: " + field)
    submitted = decisions_document["decisions"]
    if not isinstance(submitted, list):
        raise ValueError("Adjudication decisions must be an array")
    decisions, original_decisions = {}, {}
    for decision in submitted:
        if not isinstance(decision, dict) or set(decision) != _ROW_FIELDS | {"first_review", "second_review"}:
            raise ValueError("Adjudication rows must contain standard labels and unchanged review evidence")
        key = row_key(decision)
        if key in decisions:
            raise ValueError("Duplicate adjudication boundary identity")
        if key not in disputed:
            raise ValueError("Unexpected adjudication boundary identity: " + str(key))
        for field in IDENTITY_FIELDS:
            if _canonical_json(decision[field]) != _canonical_json(first[key][field]):
                raise ValueError("Adjudication identity changed: " + field)
        for field, evidence in (("first_review", first_raw[key]), ("second_review", second_raw[key])):
            if _canonical_json(decision[field]) != _canonical_json(_label_evidence(evidence)):
                raise ValueError("Adjudication original review evidence changed: " + field)
        label = {field: decision[field] for field in IDENTITY_FIELDS + LABEL_FIELDS}
        if (label["review_status"] != "pending" and
                (not isinstance(label["review_notes"], str) or not label["review_notes"].strip())):
            raise ValueError("Completed adjudication requires a nonempty human reason")
        audio = metadata["audio_sources"][key[0]]
        try:
            decisions[key] = validate_label(first[key], label, audio["num_samples"])
        except ValueError as error:
            raise ValueError("Adjudication " + str(key) + ": " + str(error)) from None
        original_decisions[key] = _label_evidence(decision)
    if set(decisions) != set(disputed):
        raise ValueError("Adjudication coverage must contain every disputed boundary, including pending decisions")
    rows, provenance, unresolved, ambiguous = [], [], [], []
    counts = {"total": len(first), "reviewed": 0, "ambiguous": 0, "pending": 0,
              "confirmed": 0, "adjudicated_reviewed": 0, "adjudicated_ambiguous": 0,
              "unresolved_adjudication": 0, "incomplete_reviews": 0}
    for key, row in first.items():
        if key in decisions:
            resolved = decisions[key]
            if resolved["review_status"] == "pending":
                method = "unresolved_adjudication"
                unresolved.append({"segment_id": key[0], "unit_index": key[1],
                                   "reason": "awaiting_human_adjudication"})
                counts["unresolved_adjudication"] += 1
            else:
                method = "human_adjudication"
                counts["adjudicated_" + resolved["review_status"]] += 1
        elif row["review_status"] == "pending" or second[key]["review_status"] == "pending":
            resolved, method = _pending(row), "incomplete_reviews"
            unresolved.append({"segment_id": key[0], "unit_index": key[1],
                               "reason": "incomplete_blind_reviews"})
            counts["incomplete_reviews"] += 1
        else:
            resolved, method = row, "retain_first_review_with_second_confirmation"
            counts["confirmed"] += 1
        counts[resolved["review_status"]] += 1
        if resolved["review_status"] == "ambiguous":
            ambiguous.append({"segment_id": key[0], "unit_index": key[1]})
        rows.append(deepcopy(resolved))
        provenance.append({"segment_id": key[0], "unit_index": key[1],
                           "first_review": _label_evidence(first_raw[key]),
                           "second_review": _label_evidence(second_raw[key]), "method": method,
                           "adjudication": deepcopy(original_decisions.get(key))})
    return dict(metadata, protocol="boundary_reference", rows=rows, provenance=provenance,
                counts=counts, unresolved=unresolved, excluded_ambiguous=ambiguous,
                status="available" if counts["reviewed"] else "unavailable")
