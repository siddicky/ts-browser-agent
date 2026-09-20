"""The observed game state, read from the page in one atomic pass.

Wordle's DOM is a known quantity — the board is rows of tiles whose `data-state`
attribute carries the feedback, the keyboard carries per-letter color states, and modals
are native `<dialog>` elements — so nothing here needs a model's judgment. `WordleState`
is what the tools return as their artifact and what `decision.py` turns into TypeSafe
request state. Click candidates still come from the generic indexed snapshot (the node
map is what makes `click` work); this module merges the two reads into one object.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel

from wordle_solver.snapshot import Element, Snapshot
from wordle_solver.wordlist import ABSENT, CORRECT, PRESENT

TileState = Literal["empty", "tbd", "correct", "present", "absent"]
Phase = Literal["playing", "won", "lost"]
DialogKind = Literal["how_to_play", "stats", "login", "other"]

_REVEALED = (CORRECT, PRESENT, ABSENT)
_ANSWER_RE = re.compile(r"the answer was\s+([a-z]{5})\b", re.IGNORECASE)

# The modal sometimes renders as a native `<dialog open>` and sometimes as a plain
# div overlay (A/B variants), so both forms are detected.
# The landing screen has a "puzzle already played" variant ("Hi Wordler / Great job on
# today's puzzle") whose button reveals the finished board.
_ALREADY_PLAYED_RE = re.compile(r"great job on today|admire puzzle|check out your progress", re.IGNORECASE)
# While the landing screen or a dialog is up, most on-page buttons (hamburger, hint,
# stats, "75% off") are distractors the CLICK question should never have to choose
# between. Pruning to the buttons that unblock play mirrors the wiki-game rule filter:
# remove options by rule instead of asking a model to avoid them.
_PLAY_RE = re.compile(r"\bplay\b|start", re.IGNORECASE)
_PRIMARY_LANDING_RE = re.compile(r"\bplay\b|start|admire|your progress", re.IGNORECASE)
_DISMISS_RE = re.compile(r"close|dismiss|got it|okay|ok\b|×", re.IGNORECASE)

# One evaluate() call per observation. Board rows are light-DOM divs keyed by build
# modules (`Row-module_row__<hash>` — the hash moves between deploys, the prefix does
# not), tiles carry `data-testid="tile"`, and the keyboard keys carry `data-key`.
_STATE_JS = r"""
() => {
  const rows = [...document.querySelectorAll("[class*='Row-module_row']")].map((row) =>
    [...row.querySelectorAll("[data-testid='tile']")].map((tile) => ({
      letter: (tile.textContent || "").trim().toLowerCase(),
      state: tile.getAttribute("data-state") || "empty",
    })),
  );
  const keyboard = {};
  for (const key of document.querySelectorAll("[class*='Keyboard-module_keyboard'] button[data-key]")) {
    keyboard[key.getAttribute("data-key")] = key.getAttribute("data-state");
  }
  const dialogEl = document.querySelector("dialog[open], [class*='modalOverlay' i]");
  const dialog = dialogEl
    ? { summary: (dialogEl.textContent || "").trim().replace(/\s+/g, " ").slice(0, 400) }
    : null;
  const toastEl = document.querySelector("[id*='toaster' i] section, [class*='toast' i]");
  return {
    rows,
    keyboard,
    dialog,
    toast: toastEl ? (toastEl.textContent || "").trim().replace(/\s+/g, " ").slice(0, 160) : null,
  };
}
"""


class Tile(BaseModel):
    """One board tile: its letter (empty when untyped) and its feedback state."""

    letter: str
    state: TileState

    @property
    def revealed(self) -> bool:
        return self.state in _REVEALED


class Dialog(BaseModel):
    """An open `<dialog>`, classified by what it is (not trusted beyond that)."""

    kind: DialogKind
    summary: str


def _classify_dialog(summary: str) -> DialogKind:
    lowered = summary.lower()
    if "how to play" in lowered:
        return "how_to_play"
    if (
        "statistics" in lowered
        or "next wordle" in lowered
        or re.search(r"\bwordle \d", lowered)
        # The win screen leads with praise, not the word "statistics".
        or re.search(r"great job|genius|magnificent|impressive|splendid|solved", lowered)
    ):
        return "stats"
    if "log in" in lowered or "sign in" in lowered or "password" in lowered:
        return "login"
    return "other"


def parse_answer(text: str) -> str | None:
    """The answer word if `text` (a dialog or toast) states it, e.g. after a loss."""
    match = _ANSWER_RE.search(text)
    return match.group(1).lower() if match else None


class WordleState(BaseModel):
    """Everything one decision needs, read atomically off the page."""

    url: str
    title: str
    rows: list[list[Tile]]
    keyboard: dict[str, str | None]
    typing: str
    phase: Phase
    landing: bool
    already_played: bool = False
    dialog: Dialog | None
    toast: str | None
    elements: list[Element]
    fingerprint: str

    @property
    def board_present(self) -> bool:
        return bool(self.rows)

    def revealed_rows(self) -> list[tuple[str, list[str]]]:
        """Submitted rows as `(guess, feedback)` pairs, in play order."""
        return [
            ("".join(tile.letter for tile in row), [tile.state for tile in row])
            for row in self.rows
            if len(row) == 5 and all(tile.revealed for tile in row)
        ]

    def active_row(self) -> list[Tile] | None:
        """The row currently accepting letters, or `None` when the board is full."""
        for row in self.rows:
            if len(row) == 5 and not all(tile.revealed for tile in row):
                return row
        return None

    @property
    def revealing(self) -> bool:
        """Whether the active row is mid-reveal: some tiles judged, some not yet.

        The flip animation judges tiles left to right over ~1.7 s, so a submitted row
        passes through states that look like half-typed input. "Typing" is only real
        when no tile in the row has been judged yet.
        """
        row = self.active_row()
        if row is None:
            return False
        revealed = sum(1 for tile in row if tile.revealed)
        return 0 < revealed < 5

    def guess_count(self) -> int:
        return len(self.revealed_rows())

    def accepts_input(self) -> bool:
        """Whether typing the next guess is possible right now."""
        return (
            self.phase == "playing"
            and self.board_present
            and self.dialog is None
            and not self.landing
            and self.typing == ""
            and not self.revealing
            and self.active_row() is not None
        )

    def click_candidates(self) -> dict[str, Element]:
        """Actionable non-keyboard buttons, keyed by their `Choice` target key."""
        return {element.target_key: element for element in self.elements if element.kind == "click"}

    def play_candidate(self) -> Element | None:
        """The landing screen's start button, when one is visible."""
        for element in self.elements:
            if element.kind == "click" and _PLAY_RE.search(element.label):
                return element
        return None

    def progress_candidate(self) -> Element | None:
        """The already-played landing's button ("Admire Puzzle"), when visible."""
        for element in self.elements:
            if element.kind == "click" and re.search(r"admire|your progress", element.label, re.IGNORECASE):
                return element
        return None

    def to_compact(self) -> dict[str, Any]:
        """The JSON-friendly shape embedded in TypeSafe request state."""
        return {
            "screen": "landing" if self.landing else "board",
            "phase": self.phase,
            "rows": [
                {"word": "".join(tile.letter for tile in row), "states": [tile.state for tile in row]}
                for row in self.rows
            ],
            "typing": self.typing,
            "dialog": self.dialog.model_dump() if self.dialog else None,
            "toast": self.toast,
        }


def phase_of(rows: list[list[Tile]]) -> Phase:
    """The observed game phase: an all-green row wins, six full rows without one lose."""
    for row in rows:
        if len(row) == 5 and all(tile.state == CORRECT for tile in row):
            return "won"
    if rows and len(rows) == 6 and all(len(row) == 5 and all(tile.revealed for tile in row) for row in rows):
        return "lost"
    return "playing"


async def aread_wordle_state(page: Any, snapshot: Snapshot) -> WordleState:
    """Read the board, keyboard, and modals; merge with the indexed element snapshot.

    Args:
        page: The Playwright page the game runs in.
        snapshot: The generic indexed snapshot taken first — it installs the node map
            `click` resolves against and supplies the click candidates.
    """
    raw = await page.evaluate(_STATE_JS)
    rows: list[list[Tile]] = []
    for raw_row in raw["rows"]:
        tiles = [Tile(letter=tile["letter"], state=tile["state"]) for tile in raw_row]
        if len(tiles) == 5:
            rows.append(tiles)
    dialog = Dialog(kind=_classify_dialog(raw["dialog"]["summary"]), summary=raw["dialog"]["summary"]) if raw["dialog"] else None
    elements = snapshot.elements
    already_played = not rows and bool(_ALREADY_PLAYED_RE.search(snapshot.text))
    if dialog is not None:
        # Only buttons that can dismiss the dialog matter while it is up.
        pruned = [el for el in elements if el.kind == "click" and _DISMISS_RE.search(el.label)]
        elements = pruned or elements
    elif not rows:
        pruned = [el for el in elements if el.kind == "click" and _PRIMARY_LANDING_RE.search(el.label)]
        elements = pruned or elements
    active = None
    for row in rows:
        if len(row) == 5 and not all(tile.revealed for tile in row):
            active = row
            break
    # Letters count as typing only while nothing in the row has been judged; during
    # the flip animation the unjudged tiles are mid-reveal, not user input.
    typing = ""
    if active is not None and not any(tile.revealed for tile in active):
        typing = "".join(tile.letter for tile in active if tile.state == "tbd")
    payload = {
        "url": snapshot.url,
        "rows": raw["rows"],
        "typing": typing,
        "dialog": raw["dialog"],
        "toast": raw["toast"],
        "elements": [(element.id, element.label) for element in snapshot.elements],
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return WordleState(
        url=snapshot.url,
        title=snapshot.title,
        rows=rows,
        keyboard=raw["keyboard"] or {},
        typing=typing,
        phase=phase_of(rows),
        landing=not rows,
        already_played=already_played,
        dialog=dialog,
        toast=raw["toast"],
        elements=elements,
        fingerprint=fingerprint,
    )


__all__ = ["Dialog", "Phase", "Tile", "TileState", "WordleState", "aread_wordle_state", "parse_answer", "phase_of"]
