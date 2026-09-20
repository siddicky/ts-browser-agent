"""A dedicated NYT Wordle solver: LangChain's `create_agent` with `langchain-typesafe` as the model.

Every turn, `WordleSolverModel` asks TypeSafe one batched request — which action to take
and, speculatively, which word to guess and which button to click — and answers with one
tool call. The code side owns everything deterministic: reading the board, the constraint
filter, and ranking candidate guesses by expected information. TypeSafe supplies the
semantic pick among the top-ranked candidates.
"""

from wordle_solver.agent import build_wordle_agent
from wordle_solver.model import WordleSolverModel
from wordle_solver.snapshot import Element, Snapshot
from wordle_solver.wordle import Dialog, Tile, WordleState

__all__ = [
    "Dialog",
    "Element",
    "Snapshot",
    "Tile",
    "WordleSolverModel",
    "WordleState",
    "build_wordle_agent",
]
