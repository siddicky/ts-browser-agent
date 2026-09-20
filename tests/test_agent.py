"""The compiled agent end to end, against a fake browser and a scripted classifier."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Iterable
from typing import Any, ClassVar

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from wordle_solver import model as model_module
from wordle_solver.agent import build_wordle_agent
from wordle_solver.browser import StalePage
from wordle_solver.decision import Decision
from wordle_solver.snapshot import Element, Snapshot
from wordle_solver.wordle import Tile, WordleState, phase_of

CLOSE = Element(id=1, role="button", kind="click", label="Close", current_value="")

NYT_URL = "https://www.nytimes.com/games/wordle/index.html"
GOAL = f"Solve today's Wordle.\n\nStart at {NYT_URL}"
STALL_WINDOW = 3


def _tiles(*specs: tuple[str, str]) -> list[Tile]:
    return [Tile(letter=letter if letter != " " else "", state=state) for letter, state in specs]


def _empty_row() -> list[Tile]:
    return _tiles(*[(" ", "empty")] * 5)


def _word_row(word: str, feedback: str) -> list[Tile]:
    names = {"c": "correct", "p": "present", "a": "absent"}
    return _tiles(*[(letter, names[code]) for letter, code in zip(word, feedback, strict=True)])


def _state(fingerprint: str, *, rows: list[list[Tile]] | None = None, elements: list[Element] | None = None) -> WordleState:
    rows = rows if rows is not None else [_empty_row() for _ in range(6)]
    return WordleState(
        url=NYT_URL,
        title="Wordle",
        rows=rows,
        keyboard={},
        typing="",
        phase=phase_of(rows),
        landing=not rows,
        dialog=None,
        toast=None,
        elements=elements or [],
        fingerprint=fingerprint,
    )


def _raw_row(word: str, feedback: str) -> list[dict[str, str]]:
    names = {"c": "correct", "p": "present", "a": "absent"}
    return [{"letter": letter, "state": names[code]} for letter, code in zip(word, feedback, strict=True)]


def _raw_empty_rows(count: int) -> list[list[dict[str, str]]]:
    return [[{"letter": "", "state": "empty"} for _ in range(5)] for _ in range(count)]


class FakePage:
    """Hands the board reader the payload for the current board version."""

    def __init__(self, boards: list[dict]) -> None:
        self._boards = boards
        self.version = 0

    async def evaluate(self, _js: str, _arg: object = None) -> dict:
        index = min(self.version, len(self._boards) - 1)
        return self._boards[index]


class FakeBrowser:
    """A page whose board advances on every action unless it is `static`."""

    created: ClassVar[list[FakeBrowser]] = []

    def __init__(self, url: str, *, static: bool = False, fail_targets: Iterable[int] = ()) -> None:
        self.url = url
        self.static = static
        self.fail_targets = set(fail_targets)
        self.acts: list[tuple[str, int | None, str | None]] = []
        self.words: list[str] = []
        self.closed = False
        self.page = FakePage(
            [
                {"rows": [], "keyboard": {}, "dialog": None, "toast": None},  # landing
                {"rows": _raw_empty_rows(6), "keyboard": {}, "dialog": None, "toast": None},  # fresh
                {
                    "rows": [_raw_row("crane", "aaaaa"), *_raw_empty_rows(5)],
                    "keyboard": {},
                    "dialog": None,
                    "toast": None,
                },
                {
                    "rows": [_raw_row("crane", "aaaaa"), _raw_row("slate", "ccccc"), *_raw_empty_rows(4)],
                    "keyboard": {},
                    "dialog": None,
                    "toast": None,
                },
            ]
        )
        FakeBrowser.created.append(self)

    async def observe(self) -> Snapshot:
        return Snapshot(
            url=self.url,
            title="Wordle",
            text="",
            elements=[CLOSE],
            can_scroll_down=False,
            can_scroll_up=False,
            fingerprint=f"fp{self.page.version}",
        )

    async def act(self, *, kind: str, node_id: int | None = None) -> None:
        if node_id in self.fail_targets:
            message = "Element is no longer actionable: occluded."
            raise StalePage(message)
        self.acts.append((kind, node_id, None))
        if not self.static and kind != "wait":  # waiting never changes the page
            self.page.version += 1

    async def type_word(self, word: str) -> bool:
        self.words.append(word)
        if not self.static:
            self.page.version += 1
        return True

    async def clear_row(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


def _factory(calls: list[dict[str, Any]] | None = None, **options: Any) -> Any:
    async def create(url: str, *, headless: bool = False, allow_private: bool = False, auth_file: object = None) -> FakeBrowser:
        if calls is not None:
            calls.append({"url": url, "headless": headless, "allow_private": allow_private, "auth_file": auth_file})
        return FakeBrowser(url, **options)

    return create


def _script(monkeypatch: pytest.MonkeyPatch, decisions: Iterable[Decision]) -> None:
    scripted = iter(decisions)

    async def fake(state: WordleState, guesses: list, candidates_left: int, constraints: str, history: list[str]) -> Decision:
        return next(scripted)

    monkeypatch.setattr(model_module, "adecide", fake)


def _run(agent: Any, goal: str = GOAL) -> list[Any]:
    result = asyncio.run(agent.ainvoke({"messages": [HumanMessage(goal)]}))
    return list(result["messages"])


@pytest.fixture(autouse=True)
def _reset_browsers() -> None:
    FakeBrowser.created.clear()


def test_a_full_solve_clicks_plays_two_guesses_and_ends_done(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(
        monkeypatch,
        [
            Decision(action="CLICK", target=CLOSE, confidence=0.9),  # the landing Play button
            Decision(action="SUBMIT_GUESS", word="crane", confidence=0.8),
            Decision(action="SUBMIT_GUESS", word="slate", confidence=0.8),
        ],
    )
    agent = build_wordle_agent(browser_factory=_factory())
    messages = _run(agent)
    browser = FakeBrowser.created[0]
    assert browser.acts == [("click", 1, None)]
    assert browser.words == ["crane", "slate"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert all(isinstance(m.artifact, WordleState) for m in tool_messages)
    assert messages[-1].content == "DONE: solved slate in 2/6."
    assert browser.closed


def test_open_uses_the_goal_url_and_session_options(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, itertools.repeat(Decision(action="WAIT")))
    calls: list[dict[str, Any]] = []
    agent = build_wordle_agent(
        browser_factory=_factory(calls),
        headless=True,
        allow_private=True,
        auth_file=".auth/nyt-cookies.txt",
        max_steps=4,
    )
    messages = _run(agent)
    assert calls == [{"url": NYT_URL, "headless": True, "allow_private": True, "auth_file": ".auth/nyt-cookies.txt"}]
    assert messages[-1].content.startswith("STALLED")
    assert FakeBrowser.created[0].closed


def test_browser_is_closed_when_the_run_ends_done(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(
        monkeypatch,
        [
            Decision(action="CLICK", target=CLOSE, confidence=0.9),
            Decision(action="SUBMIT_GUESS", word="crane", confidence=0.8),
            Decision(action="SUBMIT_GUESS", word="slate", confidence=0.8),
        ],
    )
    _run(build_wordle_agent(browser_factory=_factory()))
    assert FakeBrowser.created[0].closed


def test_no_change_actions_stall_and_close_the_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, itertools.repeat(Decision(action="CLICK", target=CLOSE, confidence=0.9)))
    agent = build_wordle_agent(browser_factory=_factory(static=True))
    messages = _run(agent)
    browser = FakeBrowser.created[0]
    assert len(browser.acts) == STALL_WINDOW
    assert messages[-1].content.startswith("STALLED")
    assert browser.closed


def test_same_target_failures_stall_early(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, itertools.repeat(Decision(action="CLICK", target=CLOSE, confidence=0.9)))
    agent = build_wordle_agent(browser_factory=_factory(fail_targets=[1]))
    messages = _run(agent)
    browser = FakeBrowser.created[0]
    assert browser.acts == []
    failed = [m for m in messages if isinstance(m, ToolMessage) and str(m.content).startswith("failed:")]
    assert len(failed) == STALL_WINDOW
    assert messages[-1].content.startswith("STALLED")
    assert browser.closed


def test_failures_across_targets_spend_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    one = Element(id=1, role="button", kind="click", label="One", current_value="")
    two = Element(id=2, role="button", kind="click", label="Two", current_value="")
    _script(
        monkeypatch,
        itertools.cycle([
            Decision(action="CLICK", target=one, confidence=0.9),
            Decision(action="CLICK", target=two, confidence=0.9),
        ]),
    )
    agent = build_wordle_agent(browser_factory=_factory(fail_targets=[1, 2]), max_steps=6)
    messages = _run(agent)
    assert "budget" in messages[-1].content
    assert FakeBrowser.created[0].closed


def test_model_messages_describe_each_action(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(
        monkeypatch,
        [
            Decision(action="CLICK", target=CLOSE, confidence=0.9),
            Decision(action="SUBMIT_GUESS", word="crane", confidence=0.8),
            Decision(action="SUBMIT_GUESS", word="slate", confidence=0.8),
        ],
    )
    messages = _run(build_wordle_agent(browser_factory=_factory()))
    descriptions = [m.content for m in messages if isinstance(m, AIMessage) and m.content]
    assert descriptions[:3] == ["OPEN " + NYT_URL, "CLICK 'Close'", "SUBMIT_GUESS 'crane'"]
