"""`feedback`, filtering, and ranking: golden cases plus a brute-force cross-check."""

from __future__ import annotations

import pytest

from wordle_solver.wordlist import (
    ABSENT,
    CORRECT,
    PRESENT,
    answers,
    constraint_summary,
    feedback,
    filter_candidates,
    guesses,
    is_allowed,
    rank_guesses,
)


def test_feedback_exact_match_all_green() -> None:
    assert feedback("crane", "crane") == (CORRECT,) * 5


def test_feedback_splits_greens_then_yellows() -> None:
    # speed vs erase: the answer's s and both e's are all unclaimed when the yellows
    # are granted; p and d never appear in the answer.
    assert feedback("speed", "erase") == (PRESENT, ABSENT, PRESENT, PRESENT, ABSENT)


def test_feedback_duplicate_letters_exhaust_the_answer() -> None:
    # allee vs apple: the first l is present, the second l has no unclaimed l left in
    # the answer and is absent, and the trailing e greens.
    assert feedback("allee", "apple") == (CORRECT, PRESENT, ABSENT, ABSENT, CORRECT)


def test_filter_matches_brute_force_simulation() -> None:
    revealed = [("crane", list(feedback("crane", "slate"))), ("mould", list(feedback("mould", "slate")))]
    expected = tuple(
        word for word in answers() if feedback("crane", word) == tuple(revealed[0][1]) and feedback("mould", word) == tuple(revealed[1][1])
    )
    assert filter_candidates(answers(), revealed) == expected


def test_filter_keeps_everything_on_an_empty_board() -> None:
    assert len(filter_candidates(answers(), [])) == len(answers())


def test_rank_prefers_informative_openers() -> None:
    ranked = rank_guesses(answers(), top_k=8)
    assert len(ranked) == 8
    # A good opener leaves well under half the answer list on average.
    assert ranked[0].expected_remaining < len(answers()) / 2
    assert all(is_allowed(guess.word) for guess in ranked)


def test_rank_single_candidate_is_that_candidate() -> None:
    ranked = rank_guesses(("crane",))
    assert ranked[0].word == "crane"
    assert ranked[0].expected_remaining == 1.0
    assert ranked[0].is_possible_answer is True


def test_rank_flags_which_candidates_are_answers() -> None:
    candidates = ("crane", "slate")
    ranked = rank_guesses(candidates, top_k=8)
    by_word = {guess.word: guess for guess in ranked}
    assert by_word["crane"].is_possible_answer is True
    assert by_word["slate"].is_possible_answer is True
    if len(ranked) > 2:
        outside = [guess for guess in ranked if guess.word not in candidates]
        assert all(guess.is_possible_answer is False for guess in outside)


def test_is_allowed_matches_the_nyt_dictionary() -> None:
    assert is_allowed("crane")
    assert is_allowed("CRANE")  # case-insensitive
    assert not is_allowed("qqqqq")
    assert not is_allowed("banana")  # six letters


def test_guess_list_contains_the_answer_list() -> None:
    assert set(answers()) <= set(guesses())


@pytest.mark.parametrize(
    ("revealed", "expected_parts"),
    [
        ([], ["No feedback"]),
        (
            [("crane", (CORRECT, ABSENT, ABSENT, ABSENT, CORRECT))],
            ["c in position 1", "e in position 5", "Not in the word: a, n, r"],
        ),
        (
            [("abbey", (ABSENT, PRESENT, ABSENT, ABSENT, ABSENT))],
            ["Yellow (in the word, wrong spot): b", "Not in the word: a, e, y"],
        ),
    ],
)
def test_constraint_summary_names_each_constraint(revealed: list, expected_parts: list[str]) -> None:
    summary = constraint_summary(revealed)
    for part in expected_parts:
        assert part in summary


def test_constraint_summary_does_not_list_a_yellow_letter_as_gray() -> None:
    # b is yellow in one row and absent in another: it IS in the word, so the gray
    # list must not contradict itself by naming it.
    summary = constraint_summary(
        [
            ("abbey", (ABSENT, PRESENT, ABSENT, ABSENT, ABSENT)),
            ("braid", (ABSENT, ABSENT, ABSENT, ABSENT, ABSENT)),
        ]
    )
    assert "b" not in summary.split("Not in the word:")[1]
