"""NumPy CTC Viterbi alignment with explicit frame constraints.

Only two active score rows are retained. Traceback stores one int8 transition
per cell in the active state band; a broad unconstrained band can still require
O(T * P) traceback bytes, but never a T * P floating-point score matrix.
"""
import numpy as np


class NoAlignmentPathError(ValueError):
    """Valid inputs and constraints admit no complete finite-score CTC path."""


def _integers(value, name, *, ndim, allow_empty=False):
    try:
        result = np.asarray(value)
        objects = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(name + " must contain integer indices") from None
    if result.ndim != ndim:
        raise ValueError(name + " has an invalid shape")
    if allow_empty and result.size == 0:
        return result.astype(np.int64)
    if result.dtype.kind not in "iu" or any(isinstance(item, (bool, np.bool_)) for item in objects.flat):
        raise ValueError(name + " must contain integers, not booleans or floats")
    if np.any(result > np.iinfo(np.int64).max):
        raise ValueError(name + " exceeds the supported integer range")
    return result.astype(np.int64, copy=False)


def _windows(value, name, count, frames):
    if value is None:
        return None
    windows = _integers(value, name, ndim=2)
    if (windows.shape != (count, 2) or np.any(windows[:, 0] < 0) or
            np.any(windows[:, 1] > frames) or np.any(windows[:, 0] >= windows[:, 1])):
        raise ValueError(name + " must be nonempty [start, end) frame pairs inside the audio")
    return windows


def _log_scores(value, name, ndim):
    try:
        result = np.asarray(value)
    except (TypeError, ValueError):
        raise ValueError(name + " must be a numeric array") from None
    if result.ndim != ndim or result.dtype.kind not in "fiu":
        raise ValueError(name + " must be a real numeric array with the required shape")
    # Negative infinity denotes a forbidden emission/entry. Normalization is
    # the caller's responsibility; finite additive scores are accepted.
    if np.isnan(result).any() or np.isposinf(result).any():
        raise ValueError(name + " cannot contain NaN or positive infinity")
    return result


def constrained_alignment(log_probs, phone_ids, word_spans, *, blank_id=40,
                          word_windows=None, onset_windows=None, silence_mask=None,
                          boundary_log_scores=None, line_start_phones=(), pause_phone_ids=()):
    """Return ordered original-frame intervals and an uncalibrated path score.

    ``word_spans`` are ordered, nonoverlapping half-open phone-index intervals;
    gaps are allowed for explicit pause phones. ``word_windows`` constrain all
    phone occupancy in a word; ``onset_windows`` constrain only its first phone's
    first occupied frame. All frame windows and returned intervals are half-open.
    A phone may stay arbitrarily long inside its occupancy window. Repeated
    adjacent phone IDs must pass through a CTC blank to remain separate tokens.

    Silence allows blank and explicitly named ``pause_phone_ids`` only.
    ``boundary_log_scores[T]`` are added once, on first entering each specified
    ``line_start_phones`` index, never on a stay or any other phone.

    ``state_path`` indexes [blank, phone0, blank, phone1, ..., blank]. Ties prefer
    stay, then advance one state, then advance two; the final phone is preferred
    to trailing blank on equal final scores. ``visited_cells`` counts cells in
    the active DP band; ``traceback_cells`` counts retained int8 traceback cells.
    Both counts include band cells rejected by emission/entry constraints.
    """
    scores = _log_scores(log_probs, "log_probs", 2)
    frames, classes = scores.shape
    if frames < 1 or classes < 2:
        raise ValueError("log_probs must have nonempty [frames, classes] shape")
    if (not isinstance(blank_id, (int, np.integer)) or isinstance(blank_id, (bool, np.bool_)) or
            not 0 <= blank_id < classes):
        raise ValueError("blank_id must be an integer class index")
    phones = _integers(phone_ids, "phone_ids", ndim=1)
    if not len(phones) or np.any(phones < 0) or np.any(phones >= classes) or np.any(phones == blank_id):
        raise ValueError("phone_ids must be nonempty non-blank class indices")
    phone_count = len(phones)
    spans = _integers(word_spans, "word_spans", ndim=2)
    if (spans.shape[1:] != (2,) or not len(spans) or np.any(spans[:, 0] < 0) or
            np.any(spans[:, 1] > phone_count) or np.any(spans[:, 0] >= spans[:, 1]) or
            np.any(spans[1:, 0] < spans[:-1, 1])):
        raise ValueError("word_spans must be ordered nonoverlapping [start, end) phone pairs")
    word_count = len(spans)
    occupancy = _windows(word_windows, "word_windows", word_count, frames)
    onsets = _windows(onset_windows, "onset_windows", word_count, frames)
    if silence_mask is None:
        silence = np.zeros(frames, dtype=bool)
    else:
        silence = np.asarray(silence_mask)
        if silence.shape != (frames,) or silence.dtype.kind != "b":
            raise ValueError("silence_mask must be a boolean array of shape [frames]")
    boundary = None if boundary_log_scores is None else _log_scores(
        boundary_log_scores, "boundary_log_scores", 1)
    if boundary is not None and boundary.shape != (frames,):
        raise ValueError("boundary_log_scores must have shape [frames]")
    starts = _integers(line_start_phones, "line_start_phones", ndim=1, allow_empty=True)
    if np.any(starts < 0) or np.any(starts >= phone_count) or len(set(starts.tolist())) != len(starts):
        raise ValueError("line_start_phones must contain distinct phone indices")
    pauses = _integers(pause_phone_ids, "pause_phone_ids", ndim=1, allow_empty=True)
    if (np.any(pauses < 0) or np.any(pauses >= classes) or np.any(pauses == blank_id) or
            len(set(pauses.tolist())) != len(pauses)):
        raise ValueError("pause_phone_ids must contain distinct non-blank class indices")

    # A phone has one continuous occupancy run. Bound its first and last frames,
    # then propagate order and repeated-phone blank requirements to prune bands.
    minimum = np.zeros(phone_count, dtype=np.int64)
    maximum = np.full(phone_count, frames - 1, dtype=np.int64)
    entry_end = np.full(phone_count, frames, dtype=np.int64)
    if occupancy is not None:
        for (first, last), (left, right) in zip(spans, occupancy):
            minimum[first:last] = left
            maximum[first:last] = right - 1
    if onsets is not None:
        for (first, _), (left, right) in zip(spans, onsets):
            minimum[first] = max(minimum[first], left)
            entry_end[first] = right
    for index in range(1, phone_count):
        minimum[index] = max(minimum[index], minimum[index - 1] + 1 + int(phones[index] == phones[index - 1]))
    latest_entry = np.minimum(maximum, entry_end - 1)
    for index in range(phone_count - 2, -1, -1):
        gap = 1 + int(phones[index] == phones[index + 1])
        maximum[index] = min(maximum[index], latest_entry[index + 1] - gap)
        latest_entry[index] = min(latest_entry[index], maximum[index])
    if np.any(minimum > latest_entry):
        raise NoAlignmentPathError("No alignment path: phone order and frame windows are incompatible")

    state_count = 2 * phone_count + 1
    lower = np.empty(state_count, dtype=np.int64)
    upper = np.empty(state_count, dtype=np.int64)
    lower[1::2], upper[1::2] = minimum, maximum
    lower[0], upper[-1] = 0, frames - 1
    lower[2::2] = minimum + 1
    upper[:-1:2] = latest_entry - 1
    state_labels = np.full(state_count, int(blank_id), dtype=np.int64)
    state_labels[1::2] = phones
    line_states = np.zeros(state_count, dtype=bool)
    line_states[2 * starts + 1] = True
    silence_allowed = state_labels == blank_id
    silence_allowed[1::2] = np.isin(phones, pauses)
    skip_allowed = np.zeros(state_count, dtype=bool)
    skip_allowed[3::2] = phones[1:] != phones[:-1]

    # Sorted activation/deactivation events avoid scanning all phones per frame.
    events_in, events_out = {}, {}
    for state, (left, right) in enumerate(zip(lower, upper)):
        if left <= right:
            events_in.setdefault(int(left), []).append(state)
            events_out.setdefault(int(right) + 1, []).append(state)
    active = np.zeros(state_count, dtype=bool)
    active_min, active_max = state_count, -1
    previous, previous_left = np.empty(0, dtype=np.float64), 0
    previous_first, previous_last = 0, 0
    traceback = []
    visited = 0
    for frame in range(frames):
        for state in events_out.get(frame, ()):
            active[state] = False
        for state in events_in.get(frame, ()):
            active[state] = True
            active_min, active_max = min(active_min, state), max(active_max, state)
        while active_min <= active_max and not active[active_min]:
            active_min += 1
        while active_max >= active_min and not active[active_max]:
            active_max -= 1
        left = max(active_min, previous_first)
        right = min(active_max + 1, previous_last + 3 if frame else 2)
        if right <= left:
            raise NoAlignmentPathError("No alignment path remains at frame {}".format(frame))
        states = np.arange(left, right)
        width = right - left
        visited += width
        current = np.full(width, -np.inf, dtype=np.float64)
        trace = np.full(width, -1, dtype=np.int8)
        valid = active[left:right].copy()
        if silence[frame]:
            valid &= silence_allowed[left:right]
        emission = scores[frame, state_labels[left:right]].astype(np.float64, copy=False)
        entry_bonus = np.zeros(width, dtype=np.float64)
        if boundary is not None:
            entry_bonus[line_states[left:right]] = boundary[frame]
        entry_valid = np.ones(width, dtype=bool)
        phone_states = states % 2 == 1
        entry_valid[phone_states] = frame < entry_end[states[phone_states] // 2]
        if frame == 0:
            initial = (states <= 1) & entry_valid
            current[initial] = emission[initial] + entry_bonus[initial]
            trace[initial] = 0
        else:
            for step in (0, 1, 2):
                previous_indices = states - step - previous_left
                allowed = (previous_indices >= 0) & (previous_indices < len(previous))
                if step:
                    allowed &= entry_valid
                if step == 2:
                    allowed &= skip_allowed[left:right]
                candidates = np.full(width, -np.inf, dtype=np.float64)
                candidates[allowed] = previous[previous_indices[allowed]] + emission[allowed]
                if step:
                    candidates[allowed] += entry_bonus[allowed]
                better = candidates > current
                current[better], trace[better] = candidates[better], step
        current[~valid], trace[~valid] = -np.inf, -1
        if np.isposinf(current).any():
            raise ValueError("Accumulated alignment score overflowed")
        finite = np.flatnonzero(np.isfinite(current))
        if not len(finite):
            raise NoAlignmentPathError("No alignment path remains at frame {}".format(frame))
        traceback.append((left, trace))
        previous, previous_left = current, left
        previous_first, previous_last = left + int(finite[0]), left + int(finite[-1])

    endings = [(float(previous[state - previous_left]), state) for state in (state_count - 2, state_count - 1)
               if previous_left <= state < previous_left + len(previous) and np.isfinite(previous[state - previous_left])]
    if not endings:
        raise NoAlignmentPathError("No alignment path reaches the final phone")
    score, state = max(endings, key=lambda item: item[0])
    path = np.empty(frames, dtype=np.int64)
    for frame in range(frames - 1, -1, -1):
        path[frame] = state
        left, trace = traceback[frame]
        step = int(trace[state - left])
        if step < 0:
            raise RuntimeError("Internal CTC traceback is incomplete")
        if frame:
            state -= step
    # Every state occupies one run on a monotone path; derive intervals in O(T).
    phone_frames = np.full((phone_count, 2), -1, dtype=np.int64)
    run_starts = np.r_[0, np.flatnonzero(path[1:] != path[:-1]) + 1]
    run_ends = np.r_[run_starts[1:], frames]
    for first, last in zip(run_starts, run_ends):
        state = path[first]
        if state % 2:
            phone_frames[state // 2] = first, last
    if np.any(phone_frames < 0):
        raise RuntimeError("Internal CTC traceback skipped a phone")
    word_frames = [[int(phone_frames[first, 0]), int(phone_frames[last - 1, 1])] for first, last in spans]
    return {"word_frames": word_frames, "phone_frames": phone_frames.tolist(), "score": score,
            "state_path": path.tolist(), "visited_cells": visited,
            "traceback_cells": sum(len(trace) for _, trace in traceback)}
