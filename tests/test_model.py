"""`WordleSolverModel` decides from messages alone; drive it with hand-built histories."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from wordle_solver import model as model_module
from wordle_solver.decision import Decision
from wordle_solver.model import WordleSolverModel
from wordle_solver.snapshot import Element
from wordle_solver.wordle import Dialog, Tile, WordleState, phase_of

CLOSE = Element(id=1, role="button", kind="click", label="Close", current_value="")

GOAL = HumanMessage("Solve today's Wordle.\n\nStart at https://www.nytimes.com/games/wordle/index.html")
NYT_URL = "https://www.nytimes.com/games/wordle/index.html"


def _tiles(*specs: tuple[str, str]) -> list[Tile]:
    return [Tile(letter=letter if letter != " " else "", state=state) for letter, state in specs]


def _empty_row() -> list[Tile]:
    return _tiles(*[(" ", "empty")] * 5)


def _state(
    fingerprint: str,
    *,
    rows: list[list[Tile]] | None = None,
    elements: list[Element] | None = None,
    dialog_summary: str | None = None,
    typing: str = "",
    toast: str | None = None,
) -> WordleState:
    rows = rows if rows is not None else [_empty_row() for _ in range(6)]
    dialog = Dialog(kind="stats", summary=dialog_summary) if dialog_summary else None
    return WordleState(
        url=NYT_URL,
        title="Wordle",
        rows=rows,
        keyboard={},
        typing=typing,
        phase=phase_of(rows),
        landing=not rows,
        dialog=dialog,
        toast=toast,
        elements=elements or [],
        fingerprint=fingerprint,
    )


def _step(
    tool: str,
    args: dict[str, Any],
    state: WordleState,
    *,
    content: str = "ok",
    description: str | None = None,
) -> list[BaseMessage]:
    call_id = f"call_{tool}_{state.fingerprint}"
    return [
        AIMessage(content=description or tool, tool_calls=[{"name": tool, "args": args, "id": call_id, "type": "tool_call"}]),
        ToolMessage(content=content, tool_call_id=call_id, artifact=state),
    ]


def _opened(fingerprint: str = "fp0") -> list[BaseMessage]:
    return [GOAL, *_step("open", {"url": NYT_URL}, _state(fingerprint))]


def _decide(*decisions: Decision, monkeypatch: pytest.MonkeyPatch, seen: list[list[str]] | None = None) -> None:
    scripted = iter(decisions)

    async def fake(state: WordleState, guesses: list, candidates_left: int, constraints: str, history: list[str]) -> Decision:
        if seen is not None:
            seen.append(history)
        return next(scripted)

    monkeypatch.setattr(model_module, "adecide", fake)


def _run(model: WordleSolverModel, messages: list[BaseMessage]) -> AIMessage:
    result = asyncio.run(model.ainvoke(messages))
    assert isinstance(result, AIMessage)
    return result


def test_first_turn_opens_the_url_from_the_goal() -> None:
    message = _run(WordleSolverModel(), [GOAL])
    assert message.tool_calls[0]["name"] == "open"
    assert message.tool_calls[0]["args"] == {"url": NYT_URL}


def test_goal_without_url_opens_nyt_by_default() -> None:
    message = _run(WordleSolverModel(), [HumanMessage("Solve the puzzle.")])
    assert message.tool_calls[0]["name"] == "open"
    assert message.tool_calls[0]["args"] == {"url": NYT_URL}


def test_won_board_ends_done_without_a_classifier_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(monkeypatch=monkeypatch)  # a scripted call would fail: never reached
    green = _tiles(("s", "correct"), ("l", "correct"), ("a", "correct"), ("t", "correct"), ("e", "correct"))
    gray = _tiles(("c", "absent"), ("r", "absent"), ("a", "absent"), ("n", "absent"), ("e", "absent"))
    messages = [GOAL, *_step("open", {"url": NYT_URL}, _state("fp0", rows=[gray, green, *[_empty_row() for _ in range(4)]]))]
    message = _run(WordleSolverModel(), messages)
    assert message.tool_calls == []
    assert message.content == "DONE: solved slate in 2/6."


def test_lost_board_ends_lost_and_parses_the_dialog_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(monkeypatch=monkeypatch)
    gray = _tiles(("c", "absent"), ("r", "absent"), ("a", "absent"), ("n", "absent"), ("e", "absent"))
    messages = [
        GOAL,
        *_step("open", {"url": NYT_URL}, _state("fp0", rows=[gray] * 6, dialog_summary="Statistics ... The answer was CRASH.")),
    ]
    message = _run(WordleSolverModel(), messages)
    assert message.content == "LOST: the answer was crash."


def test_stale_typed_row_is_cleared_not_waited_on(monkeypatch: pytest.MonkeyPatch) -> None:
    # Letters sitting unjudged past type_word's own settle are stale — a restored
    # session board, typically — and only clearing the row makes it playable again.
    _decide(monkeypatch=monkeypatch)
    row = _tiles(("w", "tbd"), ("o", "tbd"), ("r", "tbd"), (" ", "empty"), (" ", "empty"))
    messages = [*_opened(), *_step("type_word", {"word": "worry"}, _state("fp1", rows=[row, *[_empty_row() for _ in range(5)]], typing="wor"))]
    message = _run(WordleSolverModel(), messages)
    assert message.tool_calls[0]["name"] == "clear_row"
    assert "wor" in message.content


def test_scripted_guess_becomes_a_type_word_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(Decision(action="SUBMIT_GUESS", word="slate", confidence=0.8), monkeypatch=monkeypatch)
    message = _run(WordleSolverModel(), _opened())
    assert message.tool_calls[0]["name"] == "type_word"
    assert message.tool_calls[0]["args"] == {"word": "slate"}
    assert message.content == "SUBMIT_GUESS 'slate'"


def test_fallback_guess_is_marked_in_the_description(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(Decision(action="SUBMIT_GUESS", word="slate", confidence=0.2, fallback=True), monkeypatch=monkeypatch)
    message = _run(WordleSolverModel(), _opened())
    assert message.content == "SUBMIT_GUESS 'slate' (code fallback)"


def test_guess_outside_the_dictionary_ends_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(Decision(action="SUBMIT_GUESS", word="qqqqq"), monkeypatch=monkeypatch)
    message = _run(WordleSolverModel(), _opened())
    assert message.tool_calls == []
    assert message.content.startswith("BLOCKED")


def test_contradictory_board_degrades_by_dropping_oldest_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    # An archive answer outside the vendored list eventually makes exact filtering
    # impossible; the solver must keep playing on the newest rows, not block.
    seen: list[tuple[int, str]] = []

    async def fake(state: WordleState, guesses: list, candidates_left: int, constraints: str, history: list[str]) -> Decision:
        seen.append((candidates_left, constraints))
        return Decision(action="SUBMIT_GUESS", word="slate", confidence=0.8)

    monkeypatch.setattr(model_module, "adecide", fake)
    # crane green at c with r yellow, then crane fully gray: no answer satisfies both.
    green_c = _tiles(("c", "correct"), ("r", "present"), ("a", "absent"), ("n", "absent"), ("e", "absent"))
    all_absent = _tiles(("c", "absent"), ("r", "absent"), ("a", "absent"), ("n", "absent"), ("e", "absent"))
    rows = [green_c, all_absent, *[_empty_row() for _ in range(4)]]
    messages = [
        *_opened("fp0"),
        *_step("type_word", {"word": "crane"}, _state("fp1", rows=rows)),
    ]
    message = _run(WordleSolverModel(), messages)
    candidates_left, constraints = seen[-1]
    assert message.tool_calls[0]["args"] == {"word": "slate"}
    # The all-green row (answer crane) and the c-absent row cannot both hold; the
    # solver dropped constraints until candidates existed again.
    assert candidates_left > 0
    assert "oldest row(s) ignored" in constraints


def test_scripted_click_targets_the_element(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(Decision(action="CLICK", target=CLOSE, confidence=0.9), monkeypatch=monkeypatch)
    message = _run(WordleSolverModel(), _opened())
    assert message.tool_calls[0]["name"] == "click"
    assert message.tool_calls[0]["args"] == {"node_id": 1}
    assert message.content == "CLICK 'Close'"


def test_starter_bypasses_the_classifier_on_the_first_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(monkeypatch=monkeypatch)
    message = _run(WordleSolverModel(starter="crane"), _opened())
    assert message.tool_calls[0] == {
        "name": "type_word",
        "args": {"word": "crane"},
        "id": message.tool_calls[0]["id"],
        "type": "tool_call",
    }
    assert "starter" in message.content


def test_starter_only_applies_to_an_empty_board(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(Decision(action="SUBMIT_GUESS", word="slate"), monkeypatch=monkeypatch)
    played = [
        list(_tiles(("c", "absent"), ("r", "absent"), ("a", "absent"), ("n", "absent"), ("e", "absent"))),
        *[_empty_row() for _ in range(5)],
    ]
    messages = [*_opened("fp0"), *_step("type_word", {"word": "crane"}, _state("fp1", rows=played))]
    message = _run(WordleSolverModel(starter="crane"), messages)
    assert message.tool_calls[0]["args"] == {"word": "slate"}


def test_three_actions_that_change_nothing_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(monkeypatch=monkeypatch)
    messages = [*_opened("fp0")]
    for _ in range(3):
        messages += _step("click", {"node_id": 1}, _state("fp0", elements=[CLOSE]), description="CLICK 'Close'")
    message = _run(WordleSolverModel(), messages)
    assert message.tool_calls == []
    assert message.content.startswith("STALLED")


def test_waits_that_change_nothing_stall_too(monkeypatch: pytest.MonkeyPatch) -> None:
    # A revealing row settles in well under three long waits — a wait tail that long
    # means the board is stuck, so waits count toward the no-change streak.
    _decide(monkeypatch=monkeypatch)
    messages = [*_opened("fp0")]
    messages += _step("click", {"node_id": 1}, _state("fp0", elements=[CLOSE]))
    messages += _step("wait", {}, _state("fp0", elements=[CLOSE]))
    messages += _step("wait", {}, _state("fp0", elements=[CLOSE]))
    message = _run(WordleSolverModel(), messages)
    assert message.tool_calls == []
    assert message.content.startswith("STALLED")


def test_repeated_effective_actions_never_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(Decision(action="CLICK", target=CLOSE, confidence=0.9), monkeypatch=monkeypatch)
    messages = [*_opened("fp0")]
    for i in range(1, 6):
        messages += _step("click", {"node_id": 1}, _state(f"fp{i}", elements=[CLOSE]))
    message = _run(WordleSolverModel(), messages)
    assert message.tool_calls[0]["name"] == "click"


def test_same_target_failures_stall(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(monkeypatch=monkeypatch)
    messages = [*_opened("fp0")]
    for _ in range(3):
        messages += _step("type_word", {"word": "crane"}, _state("fp0"), content="failed: word rejected by the game")
    message = _run(WordleSolverModel(), messages)
    assert message.content.startswith("STALLED")


def test_step_budget_ends_the_run_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(monkeypatch=monkeypatch)
    messages = [*_opened("fp0")]
    for i in range(1, 3):
        messages += _step("click", {"node_id": 1}, _state(f"fp{i}", elements=[CLOSE]))
    message = _run(WordleSolverModel(max_steps=2), messages)
    assert message.tool_calls == []
    assert "budget" in message.content


def test_history_marks_actions_that_changed_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    _decide(Decision(action="WAIT"), monkeypatch=monkeypatch, seen=seen)
    messages = [
        *_opened("fp0"),
        *_step("click", {"node_id": 1}, _state("fp1", elements=[CLOSE]), description="CLICK 'Close'"),
        *_step("click", {"node_id": 1}, _state("fp1", elements=[CLOSE]), description="CLICK 'Close'"),
    ]
    _run(WordleSolverModel(), messages)
    assert seen == [["CLICK 'Close'", "CLICK 'Close' (page did not change)"]]


def test_mid_reveal_board_waits_without_typing(monkeypatch: pytest.MonkeyPatch) -> None:
    _decide(monkeypatch=monkeypatch)
    row = _tiles(("c", "absent"), ("r", "absent"), ("a", "tbd"), ("n", "tbd"), ("e", "tbd"))
    messages = [*_opened(), *_step("type_word", {"word": "crane"}, _state("fp1", rows=[row, *[_empty_row() for _ in range(5)]]))]
    message = _run(WordleSolverModel(), messages)
    assert message.tool_calls[0]["name"] == "wait"


def test_game_rejected_words_are_never_offered_again(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_guesses: list[list[str]] = []

    async def fake(state: WordleState, guesses: list, candidates_left: int, constraints: str, history: list[str]) -> Decision:
        seen_guesses.append([guess.word for guess in guesses])
        return Decision(action="SUBMIT_GUESS", word="slate", confidence=0.8)

    monkeypatch.setattr(model_module, "adecide", fake)
    played = [
        list(_tiles(("c", "absent"), ("r", "absent"), ("a", "absent"), ("n", "absent"), ("e", "absent"))),
        *[_empty_row() for _ in range(5)],
    ]
    messages = [
        *_opened("fp0"),
        # The game rejected `crane` (NYT's live dictionary dropped it); it must not
        # appear in the ranked options the next turn offers.
        *_step("type_word", {"word": "crane"}, _state("fp1", rows=played), content="failed: word rejected by the game"),
    ]
    message = _run(WordleSolverModel(), messages)
    assert seen_guesses and "crane" not in seen_guesses[0]
    assert message.tool_calls[0]["args"] == {"word": "slate"}


def test_bind_tools_is_accepted_and_sync_generate_is_refused() -> None:
    model = WordleSolverModel()
    assert model.bind_tools([]) is not None
    with pytest.raises(NotImplementedError, match="async-only"):
        model.invoke([GOAL])
