import itertools
import json

import numpy as np
import pytest

from t2l.alignment.constrained_dp import NoAlignmentPathError, constrained_alignment


def emissions(labels, classes=3):
    result = np.full((len(labels), classes), -12.0)
    result[np.arange(len(labels)), labels] = 0.0
    return result


def align(scores, phones=(0, 1), spans=((0, 1), (1, 2)), **options):
    return constrained_alignment(scores, phones, spans, blank_id=2, **options)


def test_repeated_phones_and_words_keep_separate_states_with_a_blank_between():
    result = align(emissions([0, 0, 2, 0, 0]), phones=[0, 0])
    assert result["phone_frames"] == result["word_frames"] == [[0, 2], [3, 5]]
    assert result["state_path"] == [1, 1, 2, 3, 3] and result["score"] == 0
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    with pytest.raises(NoAlignmentPathError, match="No alignment path"):
        align(emissions([0, 0]), phones=[0, 0])


def test_distinct_phones_can_advance_without_a_forced_blank_or_word_distance():
    result = align(emissions([0, 1]))
    assert result["state_path"] == [1, 3]
    assert result["word_frames"] == [[0, 1], [1, 2]]


def test_long_phone_stays_without_fixed_duration_or_onset_window_restricting_occupancy():
    scores = emissions([2, 2] + [0] * 80 + [2])
    result = align(scores, phones=[0], spans=[[0, 1]], onset_windows=[[2, 3]])
    assert result["phone_frames"] == [[2, 82]] and result["score"] == 0
    assert result["state_path"].count(1) == 80


def test_occupancy_window_applies_to_all_phones_in_a_word_and_end_is_half_open():
    result = align(emissions([0, 0, 0, 1, 1, 1]), spans=[[0, 2]], word_windows=[[1, 5]])
    assert result["word_frames"] == [[1, 5]]
    assert result["phone_frames"] == [[1, 3], [3, 5]]
    assert result["state_path"][0] == 0 and result["state_path"][-1] == 4


def test_onset_anchor_limits_only_first_word_phone_first_entry():
    result = align(emissions([0, 0, 0, 2, 1, 1]), onset_windows=[[1, 2], [4, 5]])
    assert result["word_frames"] == [[1, 3], [4, 6]]
    assert result["state_path"][0] == 0


def test_silence_forces_blank_but_explicit_pause_phone_can_occupy_silence():
    scores = emissions([2, 0, 1, 1, 0, 0])
    mask = np.array([False, False, True, True, False, False])
    with pytest.raises(NoAlignmentPathError):
        align(scores, phones=[0, 1, 0], spans=[[0, 1], [2, 3]], silence_mask=mask,
              word_windows=[[0, 2], [4, 6]], onset_windows=[[1, 2], [4, 5]])
    result = align(scores, phones=[0, 1, 0], spans=[[0, 1], [2, 3]], silence_mask=mask,
                   word_windows=[[0, 2], [4, 6]], onset_windows=[[1, 2], [4, 5]], pause_phone_ids=[1])
    assert result["phone_frames"] == [[1, 2], [2, 4], [4, 6]]
    assert result["word_frames"] == [[1, 2], [4, 6]] and result["score"] == 0


def test_silence_blank_gap_does_not_create_an_extra_phone_or_force_a_fixed_word_gap():
    scores = emissions([0, 0, 0, 1, 1, 1])
    mask = np.array([False, False, True, True, False, False])
    result = align(scores, silence_mask=mask, word_windows=[[0, 2], [4, 6]])
    assert result["state_path"] == [1, 1, 2, 2, 3, 3]
    assert result["word_frames"] == [[0, 2], [4, 6]]


def test_boundary_score_is_applied_once_at_the_designated_line_phone_entry():
    scores = emissions([0, 0, 2, 1, 1, 1, 2])
    boundary = np.array([-.1, -.2, -.3, -.4, -.5, -.6, -.7])
    base = align(scores)
    result = align(scores, boundary_log_scores=boundary, line_start_phones=[1])
    assert result["state_path"] == base["state_path"]
    assert result["score"] == pytest.approx(base["score"] - .4)
    both = align(scores, boundary_log_scores=boundary, line_start_phones=[0, 1])
    assert both["score"] == pytest.approx(base["score"] - .5)
    unused = align(scores, boundary_log_scores=boundary)
    assert unused == base


def test_line_entry_score_can_choose_an_onset_but_stays_do_not_pay_it_again():
    scores = np.zeros((5, 3))
    boundary = np.array([-100., -100., 0., -100., -100.])
    result = align(scores, phones=[0], spans=[[0, 1]], boundary_log_scores=boundary, line_start_phones=[0])
    assert result["phone_frames"] == [[2, 5]] and result["score"] == 0


def test_narrow_band_matches_full_solution_and_visits_fewer_cells():
    scores = emissions([2, 2, 0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 1, 1, 1, 2, 2, 2])
    phones, spans = [0, 1, 0, 1], [[i, i + 1] for i in range(4)]
    full = align(scores, phones=phones, spans=spans)
    narrow = align(scores, phones=phones, spans=spans,
                   word_windows=[[2, 5], [6, 9], [10, 13], [14, 17]])
    for key in ("word_frames", "phone_frames", "state_path", "score"):
        assert narrow[key] == full[key]
    assert narrow["visited_cells"] < full["visited_cells"] < len(scores) * (2 * len(phones) + 1)
    assert narrow["traceback_cells"] == narrow["visited_cells"]


def brute_force_score(scores, phones, spans, *, word_windows=None, onset_windows=None,
                      silence_mask=None, boundary_log_scores=None, line_start_phones=()):
    """Exhaustive small-path oracle independent of the banded recurrence."""
    labels = [2]
    for phone in phones:
        labels += [phone, 2]
    best = -np.inf
    for path in itertools.product(range(len(labels)), repeat=len(scores)):
        if path[0] not in (0, 1) or path[-1] not in (len(labels) - 2, len(labels) - 1):
            continue
        if any(not (after == before or after == before + 1 or
                    (after == before + 2 and after % 2 and labels[after] != labels[before]))
               for before, after in zip(path, path[1:])):
            continue
        if silence_mask is not None and any(silence_mask[t] and state % 2 for t, state in enumerate(path)):
            continue
        invalid = False
        for word, (first, last) in enumerate(spans):
            occupied = [t for t, state in enumerate(path) if state % 2 and first <= state // 2 < last]
            onset = next(t for t, state in enumerate(path) if state == 2 * first + 1)
            if word_windows is not None:
                invalid |= any(not word_windows[word][0] <= t < word_windows[word][1] for t in occupied)
            if onset_windows is not None:
                invalid |= not onset_windows[word][0] <= onset < onset_windows[word][1]
        if invalid:
            continue
        score = sum(scores[t, labels[state]] for t, state in enumerate(path))
        if boundary_log_scores is not None:
            score += sum(boundary_log_scores[next(t for t, state in enumerate(path) if state == 2 * phone + 1)]
                         for phone in line_start_phones)
        best = max(best, score)
    return best


@pytest.mark.parametrize("phones,options", [
    ([0, 1], {}), ([0, 0], {}), ([0, 1], {"word_windows": [[1, 3], [2, 5]]}),
    ([0, 0], {"onset_windows": [[0, 2], [3, 5]]}),
    ([0, 1], {"silence_mask": np.array([False, True, False, False, False])}),
    ([0, 1], {"word_windows": [[0, 3], [2, 5]], "onset_windows": [[1, 2], [3, 4]]}),
])
def test_banded_solver_score_matches_exhaustive_small_ctc_paths(phones, options):
    scores = np.random.default_rng(7).normal(-2, 1, (5, 3))
    boundary = np.array([-.6, -.4, -.3, -.8, -.1])
    all_options = dict(options, boundary_log_scores=boundary, line_start_phones=[1])
    expected = brute_force_score(scores, phones, [[0, 1], [1, 2]], **all_options)
    result = align(scores, phones=phones, **all_options)
    assert result["score"] == pytest.approx(expected)


@pytest.mark.parametrize("options", [
    {"word_windows": [[3, 5], [0, 2]]},
    {"word_windows": [[0, 2], [2, 5]], "onset_windows": [[3, 4], [2, 5]]},
    {"silence_mask": np.ones(5, dtype=bool)},
    {"boundary_log_scores": np.full(5, -np.inf), "line_start_phones": [0]},
])
def test_valid_but_unreachable_constraints_have_a_specific_error(options):
    with pytest.raises(NoAlignmentPathError):
        align(np.zeros((5, 3)), **options)


def test_negative_infinity_is_a_forbidden_emission_and_unreachable_is_not_fabricated():
    scores = np.full((3, 3), -np.inf)
    scores[0, 0], scores[1, 2], scores[2, 1] = 0, 0, 0
    result = align(scores)
    assert result["state_path"] == [1, 2, 3]
    scores[2, 1] = -np.inf
    with pytest.raises(NoAlignmentPathError):
        align(scores)


@pytest.mark.parametrize("scores", [[], np.zeros(3), np.zeros((2, 2, 2)), np.zeros((0, 3)),
                                      np.zeros((3, 1)), np.full((3, 3), np.nan),
                                      np.full((3, 3), np.inf), np.zeros((3, 3), dtype=complex),
                                      [["x"] * 3] * 3])
def test_bad_posterior_shape_or_values_are_plain_value_errors(scores):
    with pytest.raises(ValueError) as error:
        align(scores)
    assert not isinstance(error.value, NoAlignmentPathError)


@pytest.mark.parametrize("phones,spans", [
    ([], [[0, 1]]), ([0, 2], [[0, 1]]), ([0, 3], [[0, 1]]), ([0, -1], [[0, 1]]),
    ([0., 1.], [[0, 1]]), ([False, 1], [[0, 1]]), ([[0, 1]], [[0, 1]]),
    ([0, 1], []), ([0, 1], [[0, 0]]), ([0, 1], [[0, 3]]), ([0, 1], [[0, 2], [1, 2]]),
    ([0, 1], [[1, 2], [0, 1]]), ([0, 1], [[0., 1.]]), ([0, 1], [[False, 1]]),
])
def test_bad_phone_ids_or_word_spans_are_plain_value_errors(phones, spans):
    with pytest.raises(ValueError) as error:
        align(np.zeros((5, 3)), phones=phones, spans=spans)
    assert not isinstance(error.value, NoAlignmentPathError)


@pytest.mark.parametrize("options", [
    {"word_windows": [[0, 6], [0, 5]]}, {"word_windows": [[0, 0], [0, 5]]},
    {"word_windows": [[0, 5]]}, {"word_windows": [[0., 5.], [0., 5.]]},
    {"onset_windows": [[True, 5], [0, 5]]}, {"onset_windows": [[-1, 5], [0, 5]]},
    {"onset_windows": [[0, 5], [4, 4]]}, {"silence_mask": [0, 0, 0, 0, 0]},
    {"silence_mask": [False]}, {"boundary_log_scores": [0.]},
    {"boundary_log_scores": [0, 0, np.nan, 0, 0]},
    {"line_start_phones": [2]}, {"line_start_phones": [0, 0]}, {"line_start_phones": [False]},
    {"pause_phone_ids": [2]}, {"pause_phone_ids": [3]}, {"pause_phone_ids": [1, 1]},
    {"pause_phone_ids": [True]},
])
def test_bad_optional_configuration_cannot_be_mistaken_for_unreachable_input(options):
    with pytest.raises(ValueError) as error:
        align(np.zeros((5, 3)), **options)
    assert not isinstance(error.value, NoAlignmentPathError)


@pytest.mark.parametrize("blank", [True, 2., -1, 3])
def test_invalid_blank_class_is_rejected(blank):
    with pytest.raises(ValueError) as error:
        constrained_alignment(np.zeros((5, 3)), [0, 1], [[0, 1], [1, 2]], blank_id=blank)
    assert not isinstance(error.value, NoAlignmentPathError)


def test_solver_preserves_caller_arrays_and_supports_readonly_float32_input():
    scores = emissions([0, 0, 2, 1, 1]).astype(np.float32)
    original = scores.copy()
    scores.flags.writeable = False
    phones = np.array([0, 1], dtype=np.int32)
    spans = np.array([[0, 1], [1, 2]], dtype=np.int32)
    result = align(scores, phones=phones, spans=spans)
    assert np.array_equal(scores, original) and phones.tolist() == [0, 1]
    assert spans.tolist() == [[0, 1], [1, 2]] and result["score"] == 0
