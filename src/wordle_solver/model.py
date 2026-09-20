"""TypeSafe as the model in a stock `create_agent` loop, specialized for Wordle.

`create_agent` expects a chat model that emits tool calls and tools that return
observations. `WordleSolverModel` is that model, and it never generates text. Each
turn it reads the goal from the first human message and the game state from the last
tool result's `WordleState` artifact, narrows the answer list in code, asks TypeSafe
one batched question — which action, which word, which button — and returns an
`AIMessage` with exactly one tool call — or none, which ends the run. Game-over states
are observed facts, not judgments: a won or lost board terminates the run without
spending a TypeSafe request.

Everything the model needs is derived from the messages it is given, so it holds no
per-run state and one instance can serve any number of runs.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr

from wordle_solver.decision import adecide
from wordle_solver.wordle import WordleState, parse_answer
from wordle_solver.wordlist import (
    CORRECT,
    answers,
    constraint_summary,
    filter_candidates,
    is_allowed,
    rank_guesses,
)

_TOOL_FOR = {
    "SUBMIT_GUESS": "type_word",
    "CLICK": "click",
    "WAIT": "wait",
}
_URL = re.compile(r"https?://\S+")
_DEFAULT_URL = "https://www.nytimes.com/games/wordle/index.html"
_FAILED = "failed:"
_UNCHANGED = " (page did not change)"


def message_text(content: object) -> str:
    """Flatten message content to its text; some providers return a block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block["text"] for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


@dataclass(frozen=True)
class _Step:
    """One executed action and the game state it produced."""

    description: str
    tool: str
    signature: str
    failed: bool
    state: WordleState


def _as_state(artifact: object) -> WordleState | None:
    if isinstance(artifact, WordleState):
        return artifact
    if isinstance(artifact, dict):  # a checkpointer may hand the artifact back as plain data
        try:
            return WordleState.model_validate(artifact)
        except ValueError:
            return None
    return None


def _steps(messages: Sequence[BaseMessage]) -> list[_Step]:
    """Pair each tool-calling `AIMessage` with the `ToolMessage` that answered it."""
    steps: list[_Step] = []
    pending: AIMessage | None = None
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            pending = message
            continue
        if isinstance(message, ToolMessage) and pending is not None:
            state = _as_state(message.artifact)
            if state is not None:
                call = pending.tool_calls[0]
                steps.append(
                    _Step(
                        description=message_text(pending.content),
                        tool=call["name"],
                        signature=f"{call['name']}:{call['args'].get('word') or call['args'].get('node_id')}",
                        failed=message_text(message.content).startswith(_FAILED),
                        state=state,
                    )
                )
            pending = None
    return steps


def _changed(steps: list[_Step], index: int) -> bool:
    return steps[index].state.fingerprint != steps[index - 1].state.fingerprint


def _history(steps: list[_Step]) -> list[str]:
    """Descriptions of the actions after `open`, marked when an action changed nothing."""
    return [step.description + ("" if _changed(steps, i) else _UNCHANGED) for i, step in enumerate(steps) if i > 0]


def _stall(steps: list[_Step], limit: int) -> str | None:
    """Why the run should stop, or `None` while it is still making progress.

    Repeated *effective* actions are fine — dismissing two dialogs in a row is two
    identical clicks that each change the page. What stops a run is a tail of actions
    that changed nothing, waits included — a revealing row settles in well under two
    long waits, so a longer wait tail means the board is stuck — or of failures on the
    same target.
    """
    if len(steps) - 1 < limit:
        return None
    tail = list(range(len(steps) - limit, len(steps)))
    if all(steps[i].failed for i in tail) and len({steps[i].signature for i in tail}) == 1:
        return f"{limit} failed attempts on the same target"
    if all(not steps[i].failed and not _changed(steps, i) for i in tail):
        return f"{limit} actions in a row changed nothing"
    return None


def _call(name: str, args: dict[str, Any], *, content: str) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": args, "id": f"call_{uuid4().hex[:12]}", "type": "tool_call"}],
    )


def _solved_word(state: WordleState) -> str | None:
    """The winning guess if the board shows an all-green row."""
    for word, feedback in state.revealed_rows():
        if all(state_ == CORRECT for state_ in feedback):
            return word
    return None


class WordleSolverModel(BaseChatModel):
    """The Wordle-playing policy, packaged as a chat model so `create_agent` runs it as is.

    Args:
        max_steps: Actions allowed after the page is opened. Past it the run ends
            `BLOCKED`, through the normal exit so the browser is still closed.
        no_progress_limit: Consecutive no-change actions, or same-target failures,
            before the run ends `STALLED`.
        starter: A fixed first guess (e.g. `"crane"`); played without a TypeSafe call.
        top_k: How many top-ranked candidate guesses TypeSafe chooses among.
    """

    max_steps: int = Field(default=30, ge=1)
    no_progress_limit: int = Field(default=3, ge=1)
    starter: str | None = Field(default=None)
    top_k: int = Field(default=8, ge=1)
    _resolved_starter: str | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "wordle-typesafe"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Accept the agent's tools.

        Actions map to a fixed tool set, so the schemas are not consulted, but
        `create_agent` binds tools on every call and the base implementation raises.
        """
        return self.bind(tools=list(tools), **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = "WordleSolverModel is async-only; run the agent with `ainvoke` or `astream`."
        raise NotImplementedError(message)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = await self._next(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _next(self, messages: list[BaseMessage]) -> AIMessage:
        goal = next((message_text(m.content) for m in messages if isinstance(m, HumanMessage)), "")
        steps = _steps(messages)
        if not steps:
            match = _URL.search(goal)
            url = match.group(0).rstrip(".,;:)") if match else _DEFAULT_URL
            return _call("open", {"url": url}, content=f"OPEN {url}")
        if len(steps) - 1 >= self.max_steps:
            return AIMessage(content=f"BLOCKED: step budget of {self.max_steps} exhausted.")
        stalled = _stall(steps, self.no_progress_limit)
        if stalled is not None:
            return AIMessage(content=f"STALLED: {stalled}.")

        state = steps[-1].state
        history = _history(steps)

        # Game over is an observed board fact, not a judgment to spend a request on.
        if state.phase == "won":
            word = _solved_word(state) or "?"
            return AIMessage(content=f"DONE: solved {word} in {state.guess_count()}/6.")
        if state.phase == "lost":
            answer = None
            if state.dialog is not None:
                answer = parse_answer(state.dialog.summary)
            return AIMessage(content=f"LOST: the answer was {answer or 'not revealed'}.")
        if steps[0].state.already_played and state.phase == "playing":
            # NYT restores an in-progress board for a signed-in account that has
            # today's puzzle recorded; nothing productive is left to do.
            return AIMessage(content="BLOCKED: today's puzzle was already played on this account.")

        # Tiles mid-typing or mid-flip mean the previous submit is still being judged.
        if state.typing or state.revealing:
            return _call("wait", {}, content="WAIT — a guess is mid-reveal")

        # Starting the game from the landing screen is a launch fact, not a judgment:
        # click Play as soon as the interstitial shows it. On the already-played
        # variant ("Hi Wordler — great job"), click through to the finished board so
        # the run can report the result.
        if state.landing:
            play = state.play_candidate()
            if play is not None:
                return _call("click", {"node_id": play.id}, content=f"CLICK {play.label!r}")
            if state.already_played:
                target = state.progress_candidate() or next(iter(state.click_candidates().values()), None)
                if target is not None:
                    return _call("click", {"node_id": target.id}, content=f"CLICK {target.label!r} (finished board)")

        if state.accepts_input() and state.guess_count() == 0 and self.starter:
            return _call("type_word", {"word": self.starter}, content=f"SUBMIT_GUESS {self.starter!r} (starter)")

        revealed = state.revealed_rows()
        candidates = filter_candidates(answers(), revealed)
        # Words the game itself rejected are not in NYT's live dictionary even if the
        # vendored list disagrees — never offer them again.
        rejected = {
            step.signature.split(":", 1)[1]
            for step in steps
            if step.failed and step.tool == "type_word" and ":" in step.signature
        }
        ranked_pool = rank_guesses(candidates, top_k=self.top_k + len(rejected))
        ranked = [guess for guess in ranked_pool if guess.word not in rejected][: self.top_k]
        constraints = constraint_summary(revealed)

        decision = await adecide(state, ranked, len(candidates), constraints, history)
        if decision.action == "SUBMIT_GUESS":
            word = decision.word or ""
            if not is_allowed(word) or word in rejected:
                return AIMessage(content=f"BLOCKED: proposed guess {word!r} is not playable.")
            suffix = " (code fallback)" if decision.fallback else ""
            return _call("type_word", {"word": word}, content=f"SUBMIT_GUESS {word!r}{suffix}")
        if decision.action == "CLICK" and decision.target is not None:
            element = decision.target
            return _call("click", {"node_id": element.id}, content=f"CLICK {element.label!r}")
        if decision.action == "WAIT":
            return _call("wait", {}, content="WAIT")
        return AIMessage(content="BLOCKED: no available action can progress toward solving the puzzle.")


__all__ = ["WordleSolverModel", "message_text"]
