"""Solve today's NYT Wordle — or any past puzzle from the subscriber archive.

Usage:
    uv run --env-file .env python examples/wordle.py                # headless, one solve
    uv run --env-file .env python examples/wordle.py --headed       # watch it play
    uv run --env-file .env python examples/wordle.py --starter crane
    uv run --env-file .env python examples/wordle.py --no-auth      # play logged out
    uv run --env-file .env python examples/wordle.py --date 2026-09-19 --headed

NYT cookies from `.auth/nyt-cookies.txt` are used when present, so solves record on the
signed-in account; `--no-auth` plays anonymously (and cannot reach the archive, which is
subscriber-only). Each run opens a fresh browser, so replays never collide with stored
daily progress.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date as date_type
from pathlib import Path

from _trace import run_and_print

from wordle_solver.agent import build_wordle_agent
from wordle_solver.auth import DEFAULT_COOKIE_FILE

_URL = "https://www.nytimes.com/games/wordle/index.html"


def _goal(url: str) -> str:
    return (
        "Solve this Wordle: find the hidden five-letter word in six guesses or fewer. "
        "You win with an all-green row; make guesses that are most likely to identify "
        "the answer quickly.\n\n"
        f"Start at {url}"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headed", action="store_true", help="Show the browser window while solving.")
    parser.add_argument("--url", default=_URL, help="Any Wordle page (e.g. a clone with the same markup).")
    parser.add_argument(
        "--date",
        type=date_type.fromisoformat,
        help="Play an archive puzzle (subscriber-only): YYYY-MM-DD. Overrides --url.",
    )
    parser.add_argument("--starter", default=None, help="A fixed first guess, e.g. crane.")
    parser.add_argument("--no-auth", action="store_true", help="Ignore the NYT cookie file and play logged out.")
    parser.add_argument("--auth", type=Path, default=DEFAULT_COOKIE_FILE, help="Path to the NYT cookie file.")
    parser.add_argument("--max-steps", type=int, default=30, help="Action budget after the page opens.")
    args = parser.parse_args()

    url = f"https://www.nytimes.com/games/wordle/{args.date.isoformat()}" if args.date else args.url
    auth_file = None if args.no_auth else (args.auth if Path(args.auth).exists() else None)
    agent = build_wordle_agent(
        max_steps=args.max_steps,
        headless=not args.headed,
        auth_file=auth_file,
        starter=args.starter,
    )
    await run_and_print(agent, _goal(url))


if __name__ == "__main__":
    asyncio.run(main())
