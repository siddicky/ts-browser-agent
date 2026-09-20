"""NYT session cookies as a Playwright storage state.

The solver plays logged-out just fine, but a signed-in run records solves and streaks on
the player's account. Cookies arrive as one `Cookie:` header line (copied from a real
browser session) in a gitignored file; this module parses that into the storage-state
shape `browser.new_context(storage_state=...)` accepts. The file never leaves the
machine and its values never pass through source or logs.

The user agent must match the session the cookies were minted in — DataDome binds its
token to the fingerprint that requested it, and a mismatched UA is the fastest way to
get challenged.
"""

from __future__ import annotations

import json
from pathlib import Path

NYT_DOMAIN = ".nytimes.com"

# The UA of the browser session the cookies came from (an embedded-Chromium client).
NYT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) ZCode/3.12.3 Chrome/146.0.7680.80 Electron/41.0.3 Safari/537.36"
)

DEFAULT_COOKIE_FILE = Path(".auth/nyt-cookies.txt")


def load_storage_state(cookie_file: Path | str = DEFAULT_COOKIE_FILE) -> dict:
    """Parse a raw `Cookie:` header line into a Playwright storage state.

    Raises:
        FileNotFoundError: If the cookie file does not exist.
        ValueError: If the file parses to no usable cookies.
    """
    path = Path(cookie_file)
    header = path.read_text().strip()
    cookies = []
    for pair in header.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        name, _, value = pair.partition("=")
        if not name.strip():
            continue
        cookies.append(
            {
                "name": name.strip(),
                "value": value.strip(),
                "domain": NYT_DOMAIN,
                "path": "/",
                "secure": True,
                "httpOnly": False,
                "sameSite": "Lax",
            }
        )
    if not cookies:
        message = f"No cookies parsed from {path}."
        raise ValueError(message)
    return {"cookies": cookies, "origins": []}


def storage_state_summary(state: dict) -> str:
    """One-line, value-free description of a storage state (safe for logs)."""
    names = sorted(cookie["name"] for cookie in state["cookies"])
    return f"{len(names)} cookies: {json.dumps(names)}"


__all__ = ["DEFAULT_COOKIE_FILE", "NYT_DOMAIN", "NYT_USER_AGENT", "load_storage_state", "storage_state_summary"]
