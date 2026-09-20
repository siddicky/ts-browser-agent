"""Build the Wordle agent: LangChain's `create_agent`, with TypeSafe as the model.

```python
import asyncio
from wordle_solver import build_wordle_agent

agent = build_wordle_agent(headless=True)
goal = "Solve today's Wordle.\\n\\nStart at https://www.nytimes.com/games/wordle/index.html"
result = asyncio.run(agent.ainvoke({"messages": [("user", goal)]}))
print(result["messages"][-1].content)  # "DONE: solved X in N/6", "LOST: ...", "BLOCKED", or "STALLED"
```

There is no loop code here. The model (`WordleSolverModel`) decides one tool call per
turn from the game state it is shown; the tools (`wordle_tools`) act and return the next
state; `create_agent` runs that until the model stops calling tools. `DONE` is an
observed all-green row, not a classifier answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from wordle_solver.model import WordleSolverModel
from wordle_solver.tools import (
    BrowserFactory,
    BrowserSession,
    CloseBrowser,
    SnapshotFilter,
    wordle_tools,
)


def build_wordle_agent(
    *,
    max_steps: int = 30,
    headless: bool = False,
    allow_private: bool = False,
    auth_file: Path | str | None = None,
    starter: str | None = None,
    top_k: int = 8,
    snapshot_filter: SnapshotFilter | None = None,
    browser_factory: BrowserFactory | None = None,
    session: BrowserSession | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the solver graph.

    Args:
        max_steps: Actions allowed after the page is opened.
        headless: Whether to launch Chromium headless.
        allow_private: Allow the opened URL to resolve to a private address.
        auth_file: Optional NYT cookie file; solves are recorded on that account.
        starter: A fixed first guess; skipped TypeSafe's word pick on turn one.
        top_k: How many top-ranked candidate guesses TypeSafe chooses among.
        snapshot_filter: Optional hook applied to every element snapshot.
        browser_factory: Replaces `AsyncBrowser.create`; tests pass a fake.
        session: An existing session to act through (one run at a time per session).
    """
    if session is None:
        session = BrowserSession(
            headless=headless,
            allow_private=allow_private,
            auth_file=auth_file,
            snapshot_filter=snapshot_filter,
            browser_factory=browser_factory,
        )
    return create_agent(
        model=WordleSolverModel(max_steps=max_steps, starter=starter, top_k=top_k),
        tools=wordle_tools(session),
        middleware=[CloseBrowser(session)],
    )


__all__ = ["build_wordle_agent"]
