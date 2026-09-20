# wordle-solver

A dedicated NYT Wordle solver, built on
[`langchain-typesafe`](https://docs.langchain.com/oss/python/integrations/providers/typesafe)
and the LangChain SDK. No third-party browser-agent package. The loop is LangChain's
`create_agent`; TypeSafe is the model, and the game actions are the tools.

The division of labor is the point: **code owns everything deterministic** — reading the
board DOM, Wordle's constraint rules (greens, yellows, duplicate-letter counting),
filtering the 2,314-answer list, and ranking guesses by expected remaining candidates.
**TypeSafe owns the one semantic judgment per turn** — which of the top-ranked candidate
words is worth playing next — asked in a single batched request alongside speculative
questions for which button to click. Game-over is an observed all-green row, not a
model answer, so a win ends the run without spending a request.

## Quickstart

1. Install:

   ```bash
   uv sync
   uv run playwright install chromium
   ```

2. Add a `.env` based on `.env.example` with your `TYPESAFE_API_KEY` (no other keys are
   needed — the solver uses no chat model).
3. Solve today's puzzle:

   ```bash
   uv run --env-file .env python examples/wordle.py --headed   # watch it play
   ```

   Drop `--headed` for a headless run. Useful flags:

   - `--starter crane` — force a fixed opening guess.
   - `--no-auth` — play logged out.
   - `--url URL` — point at any Wordle with the same markup (e.g. a clone).

Each run opens a fresh browser, so replays never collide with stored daily progress.

## Playing on your NYT account

Logged-out runs solve fine but record nothing. To have solves and streaks count, drop a
raw `Cookie:` header line from a signed-in nytimes.com session into
`.auth/nyt-cookies.txt` (gitignored — see `.gitignore`). The solver installs those
cookies and pins the matching user agent, since DataDome ties its anti-bot token to the
fingerprint that minted it.

## LangSmith Studio

```bash
uv run langgraph dev
```

loads the `wordle` graph for watching decisions one tool call at a time; it holds one
browser, so run one puzzle at a time.

## How a run works

One step, one batched TypeSafe request:

1. The `open` tool launches Chromium at the NYT Wordle page (an indexed element snapshot
   plus a board read come back as the tool result's artifact).
2. `WordleSolverModel` reads the `WordleState`: rows of tiles with `correct`/`present`/
   `absent` feedback, keyboard colors, any dialog, any half-typed row.
3. Observed facts short-circuit first: a half-typed row means the previous guess is
   mid-reveal, so it waits; a won or lost board ends the run (`DONE: solved X in N/6`
   or `LOST: the answer was X`).
4. Otherwise the code side filters the answer list against every revealed row and ranks
   candidate guesses by expected remaining candidates, then TypeSafe answers three
   questions at once — `action` (submit a guess / click a button / wait / blocked),
   speculatively `guess` (one of the top-ranked words, with each candidate's
   `expected_remaining` and `possible_answer` flag as criteria), and speculatively
   `click_target` (a dialog's close button, the landing screen's play button). Only the
   head matching the answered action is read; a `guess` answer outside the offered
   options falls back to the code-ranked best word.
5. The tool acts — `type_word` clears any stale row, types, submits, and polls until the
   row's tiles leave the `tbd` state (~1.7 s of flip animation) — and the next state
   comes back as the artifact.

Stall guards stop wasted steps: three actions that change nothing, or three failures on
the same target, end the run `STALLED`; a 24-action budget ends it `BLOCKED`. Either
way the browser is closed by middleware on every exit path.

## Layout

```
src/wordle_solver/
  wordlist.py   # the answer/allowed lists, feedback simulation, constraint filter, ranking
  wordle.py     # WordleState + the one-evaluate board/keyboard/dialog reader
  browser.py    # Playwright execution: re-validated clicks, type_word with settle polling
  decision.py   # the batched TypeSafe request (action + speculative guess/click_target)
  model.py      # WordleSolverModel: the policy as a BaseChatModel
  tools.py      # open/click/type_word/wait + BrowserSession + CloseBrowser middleware
  agent.py      # build_wordle_agent: create_agent with no loop code
  auth.py       # NYT cookie file → Playwright storage state
  snapshot.py   # generic indexed DOM snapshot (installs the click node map)
  safety.py     # SSRF guard for the opened URL
data/           # wordle-answers.txt (2,314) + wordle-allowed-guesses.txt (10,656)
```

## Tests

Everything runs offline against fakes — no browser, no network, no TypeSafe request:

```bash
uv run pytest
uv run ruff check src tests examples
uv run mypy src
```
