"""Validate independent boundary labels without turning candidates into truth."""
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import math


IDENTITY_FIELDS = ("segment_id", "song_id", "language", "unit_index", "text",
                   "sample_rate", "audio_path")
LABEL_FIELDS = ("human_truth_sample", "human_truth_seconds", "confidence",
                "review_status", "ambiguous", "reviewer", "reviewed_at", "review_notes")
ONSET_DEFINITION = (
    "First audible performed phoneme onset on the original WAV timeline, "
    "including identifiable unvoiced consonant noise, aspiration or release; "
    "not necessarily the first periodic vowel cycle. Unstable boundaries are ambiguous."
)


def row_key(row):
    segment_id, index = row.get("segment_id"), row.get("unit_index")
    if not isinstance(segment_id, str) or not segment_id:
        raise ValueError("Missing segment_id")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("unit_index must be a nonnegative integer")
    return segment_id, index


def validate_label(original, label, num_samples, *, stamp=False):
    """Return one label. Identity and original-sample time are mandatory contracts."""
    if not isinstance(label, dict) or set(label) - set(IDENTITY_FIELDS + LABEL_FIELDS):
        raise ValueError("Unexpected label fields")
    for field in IDENTITY_FIELDS:
        if field in label and label[field] != original[field]:
            raise ValueError("Label identity changed: " + field)
    row = {field: original[field] for field in IDENTITY_FIELDS}
    row.update({field: label.get(field) for field in LABEL_FIELDS})
    status = row["review_status"]
    if status not in ("pending", "reviewed", "ambiguous"):
        raise ValueError("Unknown review_status")
    if status == "pending":
        if any(row[field] is not None for field in LABEL_FIELDS if field != "review_status"):
            raise ValueError("Pending rows cannot contain completed labels")
        return row
    reviewer = row["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("An actual reviewer identity is required")
    row["reviewer"] = reviewer.strip()
    if not isinstance(row["ambiguous"], bool) or row["ambiguous"] != (status == "ambiguous"):
        raise ValueError("ambiguous must agree with review_status")
    notes = row["review_notes"]
    if notes is not None and not isinstance(notes, str):
        raise ValueError("review_notes must be text")
    if status == "ambiguous":
        if not notes or not notes.strip():
            raise ValueError("Ambiguous labels need an explanation")
        if row["human_truth_sample"] is not None or row["human_truth_seconds"] is not None:
            raise ValueError("Ambiguous labels must not force a single ground-truth point")
    else:
        sample = row["human_truth_sample"]
        if not isinstance(sample, int) or isinstance(sample, bool) or not 0 <= sample < num_samples:
            raise ValueError("Human onset sample must be inside the original WAV")
        seconds = row["human_truth_seconds"]
        rate = original["sample_rate"]
        if seconds is not None:
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or
                    not math.isfinite(seconds) or seconds < 0):
                raise ValueError("Human seconds must be finite and nonnegative")
            equivalent = int((Decimal(str(seconds)) * rate).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            if equivalent != sample:
                raise ValueError("Human seconds disagree with the original sample position")
        row["human_truth_seconds"] = sample / rate
    confidence = row["confidence"]
    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or
            not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("Confidence must be finite and between zero and one")
    if stamp:
        row["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    try:
        reviewed_at = datetime.fromisoformat(row["reviewed_at"])
    except (ValueError, TypeError):
        raise ValueError("reviewed_at must be an ISO timestamp with a timezone") from None
    if reviewed_at.tzinfo is None:
        raise ValueError("reviewed_at must include a timezone")
    return row


def compare_source_annotations(candidates, labels, *, second_labels=None, agreement_limit_ms=5):
    """Audit source error; this is not an alignment-system evaluation.

    Only nonambiguous independently reviewed labels enter error metrics. With a
    second reviewer, disagreements are reported and excluded until adjudicated.
    """
    if (isinstance(agreement_limit_ms, bool) or not isinstance(agreement_limit_ms, (int, float)) or
            not math.isfinite(agreement_limit_ms) or agreement_limit_ms < 0):
        raise ValueError("agreement_limit_ms must be finite and nonnegative")
    def indexed(rows):
        result = {}
        for row in rows:
            key = row_key(row)
            if key in result:
                raise ValueError("Duplicate boundary identity")
            result[key] = row
        return result
    sources, first = indexed(candidates), indexed(labels)
    second = None if second_labels is None else indexed(second_labels)
    if set(first) != set(sources) or (second is not None and set(second) != set(sources)):
        raise ValueError("Review coverage must match the candidate identities")
    errors, disagreements, missing, ambiguous, agreement_rows = [], [], [], [], []
    source_unavailable, second_pending, second_ambiguous, agreed_rows = [], [], [], []
    reviewed_count = 0
    for key, source in sources.items():
        label = first[key]
        if label.get("review_status") != "reviewed":
            (ambiguous if label.get("review_status") == "ambiguous" else missing).append(list(key))
            continue
        reviewed_count += 1
        sample, rate = label.get("human_truth_sample"), label.get("sample_rate")
        if (isinstance(sample, bool) or not isinstance(sample, int) or sample < 0 or
                isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0):
            raise ValueError("Reviewed labels need original integer sample positions and rates")
        if second is not None:
            other = second[key]
            if other.get("review_status") != "reviewed":
                (second_ambiguous if other.get("review_status") == "ambiguous"
                 else second_pending).append(list(key))
                continue
            if other.get("reviewer") == label.get("reviewer") or not other.get("reviewer") or not label.get("reviewer"):
                raise ValueError("Two independent reviews require distinct reviewer identities")
            other_sample = other.get("human_truth_sample")
            if (isinstance(other_sample, bool) or not isinstance(other_sample, int) or other_sample < 0 or
                    other.get("sample_rate") != rate):
                raise ValueError("Second review has invalid original sample coordinates")
            difference = abs(sample - other_sample) / rate * 1000
            agreement_rows.append(difference)
            if difference > agreement_limit_ms:
                disagreements.append({"segment_id": key[0], "unit_index": key[1], "difference_ms": difference})
                continue
            agreed_rows.append(difference)
        candidate = source.get("candidate_seconds")
        if (isinstance(candidate, bool) or not isinstance(candidate, (int, float)) or
                not math.isfinite(candidate) or candidate < 0):
            source_unavailable.append(list(key))
            continue
        error = (candidate - sample / rate) * 1000
        errors.append({"segment_id": key[0], "unit_index": key[1], "signed_error_ms": error,
                       "absolute_error_ms": abs(error)})
    values = sorted(row["absolute_error_ms"] for row in errors)
    def quantile(fraction):
        if not values:
            return None
        position = (len(values) - 1) * fraction
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (position - low)
    return {"schema_version": 1, "scope": "source_annotation_audit",
            "status": "available" if values else "unavailable", "onset_definition": ONSET_DEFINITION,
            "review_count": 2 if second is not None else 1,
            "independent_adjudicated_test_set": False,
            "reviewed_count": reviewed_count, "evaluated_count": len(values), "total_count": len(sources),
            "pending_or_missing": missing, "ambiguous": ambiguous, "disagreements": disagreements,
            "source_candidate_unavailable": source_unavailable,
            "second_review_pending": second_pending, "second_review_ambiguous": second_ambiguous,
            "agreement_limit_ms": agreement_limit_ms,
            "inter_reviewer_pair_count": len(agreement_rows),
            "inter_reviewer_mae_scope": "all_reviewed_pairs_including_unadjudicated_disagreements",
            "inter_reviewer_mae_ms": sum(agreement_rows) / len(agreement_rows) if agreement_rows else None,
            "inter_reviewer_agreed_pair_count": len(agreed_rows),
            "inter_reviewer_agreed_mae_ms": sum(agreed_rows) / len(agreed_rows) if agreed_rows else None,
            "mae_ms": sum(values) / len(values) if values else None,
            "median_ae_ms": quantile(.5), "p90_ms": quantile(.9), "p95_ms": quantile(.95),
            "coverage": {str(limit) + "ms": sum(v <= limit for v in values) / len(values) if values else None
                         for limit in (5, 10, 20, 50)}, "errors": errors,
            "limitations": ["Published TextGrid agreement is not system accuracy.",
                            "Reviewer identity alone does not prove independence or adjudication.",
                            "A song-grouped, isolated test set is still required for a 5ms claim."]}
