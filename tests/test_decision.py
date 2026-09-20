"""Per-step classifier construction and answer resolution, with no request sent."""

from __future__ import annotations

from langchain_typesafe.types import ChoiceAnswer

from wordle_solver.decision import build_classifier, resolve_decision
from wordle_solver.snapshot import Element
from wordle_solver.wordle import Tile, WordleState
from wordle_solver.wordlist import RankedGuess

CLOSE = Element(id=1, role="button", kind="click", label="Close", current_value="")
PLAY = Element(id=2, role="button", kind="click", label="Play", current_value="")

CRANE = RankedGuess("crane", expected_remaining=10.5, is_possible_answer=True)
SLATE = RankedGuess("slate", expected_remaining=8.2, is_possible_answer=False)


def _tiles(*specs: tuple[str, str]) -> list[Tile]:
    return [Tile(letter=letter if letter != " " else "", state=state) for letter, state in specs]


EMPTY_ROW = _tiles(*[(" ", "empty")] * 5)


def _state(
    *,
    rows: list[list[Tile]] | None = None,
    elements: list[Element] | None = None,
    dialog_summary: str | None = None,
    typing: str = "",
) -> WordleState:
    from wordle_solver.wordle import Dialog, phase_of

    rows = rows if rows is not None else [list(EMPTY_ROW) for _ in range(6)]
    dialog = Dialog(kind="how_to_play", summary=dialog_summary) if dialog_summary else None
    return WordleState(
        url="https://www.nytimes.com/games/wordle/index.html",
        title="Wordle",
        rows=rows,
        keyboard={},
        typing=typing,
        phase=phase_of(rows),
        landing=not rows,
        dialog=dialog,
        toast=None,
        elements=elements or [],
        fingerprint="fp",
    )


def _answer(choice: str, *, confidence: float = 0.9) -> ChoiceAnswer:
    return ChoiceAnswer(type="choice", choice=choice, probabilities={choice: 1.0}, confidence=confidence)


def test_fresh_board_offers_guess_without_a_target_question() -> None:
    classifier, click_candidates, words = build_classifier(_state(), [CRANE, SLATE], 200, "No feedback yet.")
    assert set(classifier.questions) == {"action", "guess"}
    assert set(classifier.questions["action"].criteria) == {"SUBMIT_GUESS", "WAIT", "BLOCKED"}  # type: ignore[union-attr]
    assert click_candidates == {}
    assert set(words) == {"crane", "slate"}


def test_dialog_offers_click_and_keeps_the_guess_speculative() -> None:
    state = _state(elements=[CLOSE], dialog_summary="How To Play ...")
    classifier, click_candidates, _ = build_classifier(state, [CRANE], 200, "digest")
    assert set(classifier.questions) == {"action", "click_target", "guess"}
    assert "SUBMIT_GUESS" not in set(classifier.questions["action"].criteria)  # type: ignore[union-attr]
    assert set(click_candidates) == {"1"}
    assert classifier.questions["click_target"].criteria == {"1": {"label": "Close", "role": "button"}}  # type: ignore[union-attr]


def test_landing_screen_disallows_submit() -> None:
    classifier, _, _ = build_classifier(_state(rows=[], elements=[PLAY]), [CRANE], 2314, "No feedback yet.")
    assert "SUBMIT_GUESS" not in set(classifier.questions["action"].criteria)  # type: ignore[union-attr]
    assert "CLICK" in set(classifier.questions["action"].criteria)  # type: ignore[union-attr]


def test_guess_criteria_carry_the_ranking_statistics() -> None:
    classifier, _, _ = build_classifier(_state(), [CRANE, SLATE], 200, "digest")
    criteria = classifier.questions["guess"].criteria  # type: ignore[union-attr]
    assert criteria["crane"] == {"expected_remaining": 10.5, "possible_answer": True}
    assert criteria["slate"]["possible_answer"] is False  # type: ignore[index]


def test_no_candidates_drops_the_guess_question() -> None:
    classifier, _, words = build_classifier(_state(), [], 0, "digest")
    assert "guess" not in classifier.questions
    assert "SUBMIT_GUESS" not in set(classifier.questions["action"].criteria)  # type: ignore[union-attr]
    assert words == {}


def test_resolve_submit_reads_the_guess_answer() -> None:
    answers = {"action": _answer("SUBMIT_GUESS"), "guess": _answer("slate", confidence=0.8)}
    decision = resolve_decision(answers, {}, {"crane": CRANE, "slate": SLATE})
    assert decision.action == "SUBMIT_GUESS"
    assert decision.word == "slate"
    assert decision.confidence == 0.8
    assert decision.fallback is False


def test_resolve_submit_falls_back_to_the_top_ranked_word() -> None:
    answers = {"action": _answer("SUBMIT_GUESS"), "guess": _answer("banana")}
    decision = resolve_decision(answers, {}, {"crane": CRANE, "slate": SLATE})
    assert decision.word == "slate"  # lowest expected_remaining wins the fallback
    assert decision.fallback is True


def test_resolve_click_reads_its_target_head() -> None:
    answers = {
        "action": _answer("CLICK", confidence=0.4),
        "click_target": _answer("1", confidence=0.95),
        "guess": _answer("crane"),
    }
    decision = resolve_decision(answers, {"1": CLOSE}, {"crane": CRANE})
    assert decision.action == "CLICK"
    assert decision.target is CLOSE
    assert decision.confidence == 0.95
    assert decision.word is None


def test_resolve_wait_carries_no_target() -> None:
    decision = resolve_decision({"action": _answer("WAIT", confidence=0.6)}, {}, {})
    assert decision.action == "WAIT"
    assert decision.target is None
    assert decision.confidence == 0.6
