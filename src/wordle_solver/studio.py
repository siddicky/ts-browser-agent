"""The graph LangSmith Studio loads via `langgraph dev` (see `langgraph.json`).

`wordle` is the solver itself; it holds one browser, so run one puzzle at a time on it.
NYT cookies are used when the gitignored `.auth/nyt-cookies.txt` file exists, so solves
record on the signed-in account.

Usage:
    uv run langgraph dev
"""

from __future__ import annotations

from pathlib import Path

from wordle_solver.agent import build_wordle_agent
from wordle_solver.auth import DEFAULT_COOKIE_FILE

_auth = DEFAULT_COOKIE_FILE if Path(DEFAULT_COOKIE_FILE).exists() else None

wordle = build_wordle_agent(headless=False, auth_file=_auth)

__all__ = ["wordle"]
