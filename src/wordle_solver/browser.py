"""Playwright-backed execution of resolved decisions.

Every click re-resolves its element from the snapshot's node-identity map and rechecks
visibility and occlusion immediately before acting, rather than trusting geometry read
during the snapshot. A `StalePage` means the page changed since the decision was made
and the caller should re-observe instead of retrying blindly.

`type_word` is Wordle-specific: it clears any half-typed row, types the word, presses
Enter, and then polls until the row's tiles leave the `tbd` (typed, unjudged) state —
the reveal animation runs roughly 1.7 seconds per row, and reading the board before it
finishes sees stale feedback.

Async throughout: the loop runs under `create_agent`, whose tool node awaits async tools
on the event loop, and Playwright's sync API cannot be driven across threads.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

from wordle_solver.auth import NYT_USER_AGENT, load_storage_state
from wordle_solver.safety import ensure_navigable
from wordle_solver.snapshot import Snapshot, aread_snapshot
from wordle_solver.wordlist import ABSENT, CORRECT, PRESENT

if TYPE_CHECKING:
    from pathlib import Path

    from playwright.async_api import Browser as PlaywrightBrowser
    from playwright.async_api import Page, Playwright

ActionKind = Literal["click", "wait"]

TypeWordResult = Literal["accepted", "rejected", "unknown"]

_REVEALED = (CORRECT, PRESENT, ABSENT)
# The toaster element persists after a toast fades, so its *text* — not its presence —
# is what distinguishes a real rejection ("Not in word list") from a stale one.
_INVALID_WORD_RE = re.compile(r"not in word list|invalid|not a word", re.IGNORECASE)

# Re-resolves a live node by the id the snapshot assigned it, then rejects the action if
# the node detached, became hidden or disabled, or something else now occupies the click
# point (a modal, a menu, a moved element) since the snapshot was read.
_RESOLVE_AND_CLICK_JS = r"""
({ id }) => {
  const el = window.__tsFastAgent?.nodes.get(id);
  if (!el || !el.isConnected) return { ok: false, reason: "detached" };
  if (el.disabled || !el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) {
    return { ok: false, reason: "not_visible" };
  }
  const fragments = [...el.getClientRects()].filter((r) => r.width > 0 && r.height > 0);
  const rect = fragments.length
    ? fragments.reduce((a, b) => (a.width * a.height >= b.width * b.height ? a : b))
    : el.getBoundingClientRect();
  const x = rect.x + rect.width / 2;
  const y = rect.y + rect.height / 2;
  if (rect.width <= 0 || rect.height <= 0 || x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) {
    return { ok: false, reason: "off_screen" };
  }
  const atPoint = document.elementFromPoint(x, y);
  if (!atPoint || !el.contains(atPoint)) return { ok: false, reason: "occluded" };
  return { ok: true, x, y };
}
"""

# The light poll `type_word` settles on: every row's tile states plus the toast text
# and dialog presence — the submitted row is tracked by index, because once it
# finishes flipping the "active row" pointer moves to the next row.
_ROW_STATES_JS = r"""
() => {
  const rows = [...document.querySelectorAll("[class*='Row-module_row']")].map((row) =>
    [...row.querySelectorAll("[data-testid='tile']")].map((tile) => tile.getAttribute("data-state") || "empty"),
  );
  const toastEl = document.querySelector("[id*='toaster' i] section, [class*='toast' i]");
  return {
    rows,
    toast: toastEl ? (toastEl.textContent || "").trim() : "",
    dialog: !!document.querySelector("dialog[open], [class*='modalOverlay' i]"),
  };
}
"""


class StalePage(RuntimeError):
    """The page changed since the decision was made; observe again before acting."""


def _check(result: dict[str, Any]) -> None:
    if not result["ok"]:
        raise StalePage(f"Element is no longer actionable: {result['reason']}.")


class AsyncBrowser:
    """Owns one Playwright page for the lifetime of a run. Build it with `create`."""

    def __init__(self, playwright: Playwright, browser: PlaywrightBrowser, page: Page) -> None:
        self._playwright = playwright
        self._browser = browser
        self.page = page

    @classmethod
    async def create(
        cls,
        url: str,
        *,
        headless: bool = False,
        allow_private: bool = False,
        auth_file: Path | str | None = None,
    ) -> AsyncBrowser:
        """Launch Chromium and open `url`, after the URL passes `ensure_navigable`.

        With `auth_file`, NYT session cookies are installed and the user agent is pinned
        to the browser family that minted them (DataDome ties its token to that
        fingerprint), so the run plays on the signed-in account.
        """
        from playwright.async_api import async_playwright

        ensure_navigable(url, allow_private=allow_private)
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=headless)
        page: Page
        if auth_file is not None:
            state = load_storage_state(auth_file)
            context = await browser.new_context(user_agent=NYT_USER_AGENT)
            await context.add_cookies(state["cookies"])
            page = await context.new_page()
        else:
            page = await browser.new_page()
        await page.goto(url, wait_until="load")
        return cls(playwright, browser, page)

    async def observe(self) -> Snapshot:
        """Read the current page into an indexed snapshot.

        Retries briefly on a transient Playwright error, since the previous action (a
        submit click, following a link) may have just triggered a navigation that is
        still settling when this is called.
        """
        from playwright.async_api import Error as PlaywrightError

        last_error: PlaywrightError | None = None
        for _ in range(10):
            try:
                return await aread_snapshot(self.page)
            except PlaywrightError as error:
                last_error = error
                await self.page.wait_for_timeout(50)
        assert last_error is not None
        raise last_error

    async def act(self, *, kind: ActionKind, node_id: int | None = None) -> None:
        """Execute one resolved action, re-validating the target immediately before it runs.

        Raises:
            StalePage: If a click target is no longer actionable.
        """
        if kind == "wait":
            await self.page.wait_for_timeout(300)
            return
        if node_id is None:  # pragma: no cover - the tools always pass one
            message = f"Action kind {kind!r} requires a target element."
            raise ValueError(message)
        result = await self.page.evaluate(_RESOLVE_AND_CLICK_JS, {"id": node_id})
        _check(result)
        await self.page.mouse.click(result["x"], result["y"])

    async def clear_row(self) -> None:
        """Backspace the active row empty, verifying — NYT drops keys while animating.

        The shake animation after a rejected guess swallows quick keystrokes, so the
        presses are spaced out and the row is re-read until it is actually empty.
        """
        for _ in range(3):
            raw = await self.page.evaluate(_ROW_STATES_JS)
            active = next((row for row in raw["rows"] if not all(state in _REVEALED for state in row)), None)
            if active is None or all(state == "empty" for state in active):
                return
            for _ in range(5):
                await self.page.keyboard.press("Backspace")
                await self.page.wait_for_timeout(120)
            await self.page.wait_for_timeout(200)

    async def type_word(self, word: str) -> TypeWordResult:
        """Type a five-letter guess, submit it, and wait for the row to be judged.

        The submitted row is tracked by index: once its flip animation finishes, the
        game's "active row" pointer moves on, so watching the active row would never
        see the verdict. A rejected word (not in NYT's list) shows a toast and leaves
        the tiles `tbd`; the row is cleared so the next decision starts clean. A stale
        toast from an earlier rejection is ignored by matching the toast's text.

        Returns:
            `"accepted"`, `"rejected"`, or `"unknown"` (no verdict within the timeout).
        """
        board = await self.page.evaluate(_ROW_STATES_JS)
        target = sum(1 for row in board["rows"] if all(state in _REVEALED for state in row))
        await self.clear_row()
        await self.page.keyboard.type(word, delay=40)
        await self.page.keyboard.press("Enter")
        for _ in range(40):  # ~6 s: five tiles' flip animation plus slack
            board = await self.page.evaluate(_ROW_STATES_JS)
            row = board["rows"][target] if target < len(board["rows"]) else None
            if row is not None and len(row) == 5 and all(state in _REVEALED for state in row):
                return "accepted"
            if board["dialog"]:
                return "accepted"  # a stats dialog after a win/loss means the guess landed
            if _INVALID_WORD_RE.search(board["toast"]) and row is not None and all(
                state in ("tbd", "empty") for state in row
            ):
                await self.clear_row()
                return "rejected"
            await self.page.wait_for_timeout(150)
        # No verdict — an unmatched rejection toast, or keys going somewhere else.
        # Leave no half-typed row behind for the next decision to trip over.
        await self.clear_row()
        return "unknown"

    async def close(self) -> None:
        await self._browser.close()
        await self._playwright.stop()


__all__ = ["ActionKind", "AsyncBrowser", "StalePage"]
