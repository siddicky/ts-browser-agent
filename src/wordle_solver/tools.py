"""Game actions as tools, and the one browser they act on.

Each tool performs its action and then observes, returning a short status line as the
tool result's content and the new `WordleState` as its artifact. That artifact is how
the model sees the game on its next turn — the ordinary tool-calling contract, with a
structured observation instead of prose.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.tools import BaseTool, tool
from langgraph.runtime import Runtime

from wordle_solver.browser import ActionKind, AsyncBrowser, StalePage
from wordle_solver.snapshot import SnapshotFilter
from wordle_solver.wordle import WordleState, aread_wordle_state
from wordle_solver.wordlist import is_allowed

BrowserFactory = Callable[..., Awaitable[AsyncBrowser]]


def _describe(state: WordleState) -> str:
    """One human-readable line about the observed game state."""
    if state.landing:
        return "landing screen (the board is not up yet)"
    parts = [f"guess {state.guess_count()}/6", f"phase={state.phase}"]
    if state.typing:
        parts.append(f"typing={state.typing!r}")
    if state.dialog is not None:
        parts.append(f"dialog={state.dialog.kind}")
    if state.toast:
        parts.append(f"toast={state.toast!r}")
    return ", ".join(parts)


class BrowserSession:
    """The browser a graph instance acts on, opened by the `open` tool.

    It lives here rather than in graph state because a Playwright handle cannot be
    checkpointed. One session means one run at a time per graph instance.

    Args:
        headless: Whether to launch Chromium headless.
        allow_private: Allow the opened URL to resolve to a private or loopback address;
            see `wordle_solver.safety.ensure_navigable`. Fixed here, never a tool
            argument.
        auth_file: Optional NYT cookie file (a raw `Cookie:` header line); installs the
            session so solves are recorded on the signed-in account.
        snapshot_filter: Optional hook applied to every element snapshot before the
            game state is built, for callers that want to drop or relabel candidates.
        browser_factory: Replaces `AsyncBrowser.create`; tests pass a fake.
    """

    def __init__(
        self,
        *,
        headless: bool = False,
        allow_private: bool = False,
        auth_file: Path | str | None = None,
        snapshot_filter: SnapshotFilter | None = None,
        browser_factory: BrowserFactory | None = None,
    ) -> None:
        self.headless = headless
        self.allow_private = allow_private
        self.auth_file = auth_file
        self.snapshot_filter = snapshot_filter
        self._factory: BrowserFactory = browser_factory or AsyncBrowser.create
        self.browser: AsyncBrowser | None = None

    async def open(self, url: str) -> None:
        await self.close()
        self.browser = await self._factory(
            url, headless=self.headless, allow_private=self.allow_private, auth_file=self.auth_file
        )

    def require(self) -> AsyncBrowser:
        if self.browser is None:
            message = "No page is open; the `open` tool must run first."
            raise RuntimeError(message)
        return self.browser

    async def observe(self) -> WordleState:
        """One atomic read: the indexed element snapshot, filtered, then the board read."""
        browser = self.require()
        snapshot = await browser.observe()
        if self.snapshot_filter is not None:
            snapshot = self.snapshot_filter(snapshot)
        return await aread_wordle_state(browser.page, snapshot)

    async def close(self) -> None:
        if self.browser is not None:
            await self.browser.close()
            self.browser = None


def wordle_tools(session: BrowserSession) -> list[BaseTool]:
    """The action tools for one session. Every tool returns `(status, WordleState)`."""

    async def report(status: str) -> tuple[str, WordleState]:
        state = await session.observe()
        return f"{status} — {_describe(state)}", state

    async def act(kind: ActionKind, node_id: int | None = None) -> tuple[str, WordleState]:
        try:
            await session.require().act(kind=kind, node_id=node_id)
        except StalePage as error:
            return await report(f"failed: {error}")
        return await report("ok")

    @tool("open", response_format="content_and_artifact")
    async def open_page(url: str) -> tuple[str, WordleState]:
        """Open a page in a fresh browser. Always the first action of a run."""
        await session.open(url)
        return await report("ok")

    @tool(response_format="content_and_artifact")
    async def click(node_id: int) -> tuple[str, WordleState]:
        """Click the element with this id from the current page's element table."""
        return await act("click", node_id)

    @tool(response_format="content_and_artifact")
    async def type_word(word: str) -> tuple[str, WordleState]:
        """Type a five-letter guess into the board and submit it."""
        if not is_allowed(word):
            return await report(f"failed: {word!r} is not a valid Wordle guess")
        try:
            outcome = await session.require().type_word(word)
        except StalePage as error:
            return await report(f"failed: {error}")
        if outcome == "accepted":
            return await report("ok")
        if outcome == "rejected":
            return await report("failed: word rejected by the game")
        return await report("failed: the board did not settle after the guess")

    @tool(response_format="content_and_artifact")
    async def wait() -> tuple[str, WordleState]:
        """Wait for the board to settle; longer while a row is mid-reveal."""
        state = await session.observe()
        if state.revealing:
            # The flip animation runs ~1.7 s; one long sleep beats several short steps.
            await asyncio.sleep(1.4)
        return await report("ok")

    return [open_page, click, type_word, wait]


class CloseBrowser(AgentMiddleware[AgentState[Any], Any, Any]):
    """Close the session's browser when the run ends, on every exit path."""

    def __init__(self, session: BrowserSession) -> None:
        super().__init__()
        self._session = session

    async def aafter_agent(self, state: AgentState[Any], runtime: Runtime[Any]) -> None:
        await self._session.close()


__all__ = ["BrowserFactory", "BrowserSession", "CloseBrowser", "SnapshotFilter", "wordle_tools"]
