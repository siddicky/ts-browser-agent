"""`aread_wordle_state` against canned page payloads — no browser, no network."""

from __future__ import annotations

from wordle_solver.snapshot import Element, Snapshot
from wordle_solver.wordle import (
    Tile,
    WordleState,
    aread_wordle_state,
    parse_answer,
    phase_of,
)

CLOSE = Element(id=1, role="button", kind="click", label="Close", current_value="")

EMPTY_ROW = [(" ", "empty")] * 5
FRESH_BOARD = [EMPTY_ROW] * 6


class FakePage:
    """Just `evaluate`, which is all the board reader uses."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def evaluate(self, _js: str, _arg: object = None) -> dict:
        return self.payload


def _pair(item: object) -> tuple[str, str]:
    """Accept both plain (letter, state) tuples and `Tile` models in raw rows."""
    if isinstance(item, Tile):
        return (item.letter or " ", item.state)
    assert isinstance(item, tuple) and len(item) == 2
    return (item[0], item[1])


def _raw(
    rows: list,
    *,
    keyboard: dict[str, str | None] | None = None,
    dialog: dict | None = None,
    toast: str | None = None,
) -> dict:
    return {
        "rows": [[{"letter": letter if letter != " " else "", "state": state} for letter, state in map(_pair, row)] for row in rows],
        "keyboard": keyboard or {},
        "dialog": dialog,
        "toast": toast,
    }


def _snapshot(elements: list[Element] | None = None) -> Snapshot:
    return Snapshot(
        url="https://www.nytimes.com/games/wordle/index.html",
        title="Wordle",
        text="",
        elements=elements or [],
        can_scroll_down=False,
        can_scroll_up=False,
        fingerprint="snap",
    )


async def _read(payload: dict, elements: list[Element] | None = None) -> WordleState:
    return await aread_wordle_state(FakePage(payload), _snapshot(elements))


def _tiles(*specs: tuple[str, str]) -> list[Tile]:
    return [Tile(letter=letter if letter != " " else "", state=state) for letter, state in specs]


async def test_landing_page_has_no_board() -> None:
    state = await _read(_raw([]))
    assert state.landing is True
    assert state.board_present is False
    assert state.phase == "playing"
    assert state.accepts_input() is False


async def test_fresh_board_accepts_input() -> None:
    state = await _read(_raw(FRESH_BOARD))
    assert state.landing is False
    assert state.phase == "playing"
    assert state.typing == ""
    assert state.guess_count() == 0
    assert state.accepts_input() is True


async def test_revealed_row_feeds_guess_count_and_phase() -> None:
    rows = [
        list(_tiles(("c", "absent"), ("r", "absent"), ("a", "absent"), ("n", "correct"), ("e", "correct"))),
        *[_tiles(*EMPTY_ROW) for _ in range(5)],
    ]
    state = await _read(_raw(rows))
    assert state.revealed_rows() == [("crane", ["absent", "absent", "absent", "correct", "correct"])]
    assert state.guess_count() == 1
    assert state.phase == "playing"
    assert state.accepts_input() is True


async def test_all_green_row_is_a_win() -> None:
    win = list(_tiles(("s", "correct"), ("l", "correct"), ("a", "correct"), ("t", "correct"), ("e", "correct")))
    state = await _read(_raw([win, *[_tiles(*EMPTY_ROW) for _ in range(5)]]))
    assert state.phase == "won"
    assert state.accepts_input() is False


async def test_six_revealed_rows_without_green_is_a_loss() -> None:
    gray = list(_tiles(("c", "absent"), ("r", "absent"), ("a", "absent"), ("n", "absent"), ("e", "absent")))
    state = await _read(_raw([gray] * 6))
    assert state.phase == "lost"


async def test_half_typed_row_is_typing_not_input() -> None:
    rows = [_tiles(("w", "tbd"), ("o", "tbd"), ("r", "tbd"), (" ", "empty"), (" ", "empty")), *[_tiles(*EMPTY_ROW) for _ in range(5)]]
    state = await _read(_raw(rows))
    assert state.typing == "wor"
    assert state.revealing is False
    assert state.accepts_input() is False


async def test_mid_reveal_row_is_not_typing_but_is_revealing() -> None:
    # A submitted row mid-flip: two tiles judged, three still `tbd`. This looks like
    # half-typed input but is the reveal animation — the model must wait, not treat
    # the leftover letters as user typing.
    rows = [
        _tiles(("c", "absent"), ("r", "absent"), ("a", "tbd"), ("n", "tbd"), ("e", "tbd")),
        *[_tiles(*EMPTY_ROW) for _ in range(5)],
    ]
    state = await _read(_raw(rows))
    assert state.typing == ""
    assert state.revealing is True
    assert state.guess_count() == 0
    assert state.accepts_input() is False


async def test_dialog_blocks_input_and_is_classified() -> None:
    state = await _read(_raw(FRESH_BOARD, dialog={"summary": "How To Play Guess the Wordle in 6 tries."}))
    assert state.dialog is not None
    assert state.dialog.kind == "how_to_play"
    assert state.accepts_input() is False


async def test_stats_dialog_kind() -> None:
    state = await _read(_raw(FRESH_BOARD, dialog={"summary": "Statistics Wordle No. 1919 3/6 Next Wordle"}))
    assert state.dialog is not None
    assert state.dialog.kind == "stats"


async def test_click_candidates_come_from_the_snapshot() -> None:
    state = await _read(_raw(FRESH_BOARD), elements=[CLOSE])
    assert set(state.click_candidates()) == {"1"}
    assert state.click_candidates()["1"].label == "Close"


async def test_fingerprint_tracks_the_board() -> None:
    fresh = await _read(_raw(FRESH_BOARD))
    played = await _read(_raw([list(_tiles(("c", "absent"), ("r", "absent"), ("a", "absent"), ("n", "absent"), ("e", "absent"))), *[_tiles(*EMPTY_ROW) for _ in range(5)]]))
    assert fresh.fingerprint != played.fingerprint


def test_phase_of_rejects_nothing_on_partial_boards() -> None:
    rows = [_tiles(*EMPTY_ROW), *[_tiles(*EMPTY_ROW) for _ in range(4)]]
    assert phase_of(rows) == "playing"


def test_parse_answer_extracts_the_revealed_word() -> None:
    assert parse_answer("Statistics ... The answer was CRASH.") == "crash"
    assert parse_answer("no answer here") is None
