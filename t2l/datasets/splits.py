"""Deterministic song-group splits with explicit reviewed or dataset identities."""
import hashlib
import json
import math
import re
from collections import Counter


def assign_song_splits(segments, song_review=None, *, seed=7,
                       ratios=None, held_out_singers=None, identity_mode="reviewed"):
    """Assign complete songs using a human review or explicit dataset identities.

    The default mode requires ``song_review`` with ``reviewed_by`` and
    ``song_ids`` (inferred -> canonical), preserving the reviewed-mode contract.
    ``identity_mode="dataset"`` uses each existing dataset ``song_id`` without
    claiming human review or resolving aliases; it must not receive a review.
    Audio annotations remain unreviewed in either mode.
    Each segment must carry its source WAV SHA-256 in ``sources.wav.sha256``.
    This function never derives whole-recording offsets or accuracy labels.
    """
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1} if ratios is None else ratios
    if (set(ratios) != {"train", "val", "test"} or
            any(isinstance(value, bool) or not isinstance(value, (float, int)) or
                not math.isfinite(value) or value < 0 for value in ratios.values()) or
            not math.isclose(sum(ratios.values()), 1.0, rel_tol=0, abs_tol=1e-9)):
        raise ValueError("train/val/test ratios must be finite, nonnegative and sum to 1")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if identity_mode not in ("reviewed", "dataset"):
        raise ValueError("identity_mode must be reviewed or dataset")
    reviewed = identity_mode == "reviewed"
    if reviewed:
        if (not isinstance(song_review, dict) or
                not isinstance(song_review.get("reviewed_by"), str) or
                not song_review["reviewed_by"].strip()):
            raise ValueError("A song identity review with reviewed_by is required")
        song_ids = song_review.get("song_ids")
        if not isinstance(song_ids, dict):
            raise ValueError("song_review.song_ids must map inferred IDs to reviewed canonical IDs")
    elif song_review is not None:
        raise ValueError("dataset identity mode cannot also receive song_review")
    segments = list(segments)
    if not segments:
        raise ValueError("Cannot split an empty dataset")

    records = {}
    audio_identities = {}
    canonical_groups = {}
    for segment in segments:
        segment_id, inferred = segment.get("segment_id"), segment.get("song_id")
        if not isinstance(segment_id, str) or not segment_id or segment_id in records:
            raise ValueError("Missing or duplicate segment_id: {!r}".format(segment_id))
        if not isinstance(inferred, str) or not inferred.strip():
            raise ValueError("Missing song_id for {!r}".format(segment_id))
        if reviewed:
            canonical = song_ids.get(inferred)
            if not isinstance(canonical, str) or not canonical.strip():
                raise ValueError("Unreviewed song identity: {!r}".format(inferred))
            canonical = canonical.strip()
        else:
            canonical = inferred
        sources = segment.get("sources")
        wav_source = sources.get("wav") if isinstance(sources, dict) else None
        digest = wav_source.get("sha256") if isinstance(wav_source, dict) else None
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("Missing or invalid sources.wav.sha256 for segment {!r}; "
                             "a 64-character hexadecimal WAV SHA-256 is required".format(segment_id))
        audio_key = digest.lower()
        if audio_key in audio_identities and audio_identities[audio_key] != canonical:
            raise ValueError("The same audio has conflicting {} song identities".format(identity_mode))
        audio_identities[audio_key] = canonical
        records[segment_id] = segment
        canonical_groups[segment_id] = canonical

    # Derived mixtures, alternate versions and paired speech can preserve the
    # source segment relation. They must never receive an unrelated identity.
    for segment_id, segment in records.items():
        parent = segment.get("parent_segment_id")
        if parent is not None:
            if parent == segment_id:
                raise ValueError("A segment cannot be its own parent")
            if parent not in records:
                raise ValueError("Unknown parent_segment_id: {!r}".format(parent))
            if canonical_groups[parent] != canonical_groups[segment_id]:
                raise ValueError("Derived segment and parent have different song identities")

    groups = sorted(set(canonical_groups.values()))
    held_out_singers = [] if held_out_singers is None else list(held_out_singers)
    if any(not isinstance(singer, str) or not singer.strip() for singer in held_out_singers):
        raise ValueError("Held-out singer identities must be nonempty strings")
    held_out_singers = sorted(set(held_out_singers))
    singers = {segment.get("singer") for segment in segments}
    if set(held_out_singers) - singers:
        raise ValueError("Unknown held-out singer")
    reserved_test = {canonical_groups[key] for key, segment in records.items()
                     if segment.get("singer") in held_out_singers}
    if reserved_test and ratios["test"] == 0:
        raise ValueError("Singer holdout requires a nonempty test split")
    if reserved_test and any(not isinstance(segment.get("singer"), str) or not segment["singer"].strip()
                             for segment in segments):
        raise ValueError("Singer holdout requires singer identities on every segment")
    active = [key for key in ("train", "val", "test") if ratios[key] > 0]
    if len(groups) < len(active):
        raise ValueError("Need at least {} {} songs for {} nonempty splits; got {}".format(
            len(active), identity_mode, len(active), len(groups)))
    # Reserve one song per active split, then approach requested ratios. This
    # guarantees useful nonempty partitions without splitting a song to fit.
    minimum_counts = {key: max(int(key in active), len(reserved_test) if key == "test" else 0)
                      for key in ("train", "val", "test")}
    if sum(minimum_counts.values()) > len(groups):
        raise ValueError("Singer holdout leaves too few independent songs for nonempty train/val splits")
    counts = dict(minimum_counts)
    for _ in range(len(groups) - sum(minimum_counts.values())):
        key = max(active, key=lambda key: (len(groups) * ratios[key] - counts[key],
                                          -active.index(key)))
        counts[key] += 1
    ranked = sorted(set(groups) - reserved_test, key=lambda group: (
        hashlib.sha256(json.dumps([seed, group], ensure_ascii=False).encode("utf8")).hexdigest(),
        group))
    assignments, cursor = {group: "test" for group in reserved_test}, 0
    for split in ("train", "val", "test"):
        count = counts[split] - (len(reserved_test) if split == "test" else 0)
        for group in ranked[cursor:cursor + count]:
            assignments[group] = split
        cursor += count
    rows = [{"segment_id": key, "canonical_song_id": canonical_groups[key],
             "split": assignments[canonical_groups[key]]} for key in sorted(records)]
    return {"schema_version": 1, "seed": seed, "requested_ratios": dict(ratios),
            "identity_mode": identity_mode,
            "song_identity_source": "human_song_review" if reviewed else "dataset_song_id",
            "song_identity_reviewed": reviewed,
            "aliases_confirmed": reviewed,
            "song_identity_reviewed_by": song_review["reviewed_by"] if reviewed else None,
            "song_counts": counts,
            "segment_counts": dict(Counter(row["split"] for row in rows)),
            "assignments": rows, "boundary_annotations_reviewed": False,
            "singer_independent": bool(held_out_singers),
            "held_out_singers": held_out_singers,
            "singer_independent_test_segment_ids": sorted(key for key, segment in records.items()
                                                         if segment.get("singer") in held_out_singers),
            "limitations": (["Song identities rely on the supplied human review."] if reviewed else [
                            "canonical_song_id contains the dataset song_id, not a human-confirmed canonical identity.",
                            "Song identities and aliases have not been verified by a human.",
                            "Unrecognized aliases or unrecorded source relations may still cross splits."]) + [
                            "Only explicitly held-out singers are excluded from train and validation; other singers may recur.",
                            "Songs shared with held-out singers are reserved entirely for test, so requested ratios may be unattainable.",
                            "This split does not make annotation boundaries accurate."]}
