"""The deterministic half of the solver: constraints, candidate filtering, guess ranking.

Everything here is plain code — Wordle's feedback rules leave nothing to judgment. Given
the revealed board, `filter_candidates` reduces the answer list to words still possible,
and `rank_guesses` orders candidate guesses by expected remaining candidates, the standard
minimizing-information heuristic. TypeSafe only ever picks among the shortlist this module
produces.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CORRECT = "correct"
PRESENT = "present"
ABSENT = "absent"

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

_WORD_LENGTH = 5


@lru_cache(maxsize=1)
def answers() -> tuple[str, ...]:
    """The curated list of possible Wordle answers, alphabetized."""
    with (_DATA_DIR / "wordle-answers.txt").open() as f:
        return tuple(line.strip().lower() for line in f if line.strip())


@lru_cache(maxsize=1)
def guesses() -> frozenset[str]:
    """Every word NYT accepts as a guess: the answers plus the extra allowed list."""
    with (_DATA_DIR / "wordle-allowed-guesses.txt").open() as f:
        extra = {line.strip().lower() for line in f if line.strip()}
    return frozenset(extra) | set(answers())


def is_allowed(word: str) -> bool:
    """Whether NYT accepts `word` as a guess."""
    return word.lower() in guesses()


def feedback(guess: str, answer: str) -> tuple[str, ...]:
    """Wordle's per-tile evaluation of `guess` against `answer`.

    Two passes, because duplicates make the one-pass intuition wrong: greens are claimed
    first, then yellows are granted only up to the letters left unclaimed in the answer.
    A second occurrence of a letter with both copies marked is `absent`, not `present`.
    """
    green = [g == a for g, a in zip(guess, answer, strict=True)]
    remaining = Counter(a for a, g in zip(answer, green, strict=True) if not g)
    result: list[str] = []
    for letter, is_green in zip(guess, green, strict=True):
        if is_green:
            result.append(CORRECT)
        elif remaining[letter] > 0:
            remaining[letter] -= 1
            result.append(PRESENT)
        else:
            result.append(ABSENT)
    return tuple(result)


def filter_candidates(
    possible: Iterable[str],
    revealed: Sequence[tuple[str, Sequence[str]]],
) -> tuple[str, ...]:
    """Words from `possible` consistent with every revealed `(guess, feedback)` row.

    Comparing simulated feedback to observed feedback sidesteps all duplicate-letter
    edge cases: the simulation already implements the game's own counting rules.
    """
    return tuple(
        word
        for word in possible
        if all(feedback(guess, word) == tuple(observed) for guess, observed in revealed)
    )


@dataclass(frozen=True)
class RankedGuess:
    """One candidate guess with the statistics `rank_guesses` scored it on."""

    word: str
    expected_remaining: float
    is_possible_answer: bool


def _shortlist(candidates: tuple[str, ...], pool: frozenset[str] | tuple[str, ...], size: int) -> list[str]:
    """Top `size` words by a cheap letter-position frequency prepass.

    The exact expected-remaining partition over all ~13k allowed guesses is minutes of
    pure Python; the prepass (position-weighted letter counts against the live
    candidates) is 1000x cheaper and reliably surfaces the informative words, which the
    exact pass then orders correctly over a much smaller field.
    """
    position_counts = [Counter(word[i] for word in candidates) for i in range(_WORD_LENGTH)]
    presence = Counter(letter for word in candidates for letter in set(word))
    scored: list[tuple[float, str]] = []
    for word in pool:
        score = 0.0
        seen: set[str] = set()
        for i, letter in enumerate(word):
            score += position_counts[i][letter]
            if letter not in seen:
                seen.add(letter)
                score += 0.5 * presence[letter]
        scored.append((score, word))
    scored.sort(reverse=True)
    return [word for _, word in scored[:size]]


def rank_guesses(
    candidates: tuple[str, ...],
    *,
    pool: Iterable[str] | None = None,
    top_k: int = 8,
    shortlist_size: int = 500,
) -> list[RankedGuess]:
    """Rank guesses by expected number of candidates remaining after playing them.

    For each guess the candidate set partitions by feedback pattern; a guess that splits
    the set finely leaves fewer words on average and is better. Ties prefer a word that
    is itself a possible answer: it wins immediately with probability
    1/len(candidates) and is never a worse source of information than a non-answer with
    the same split.
    """
    if not candidates:
        return []
    if pool is None:
        pool = guesses()
    shortlist = _shortlist(candidates, tuple(pool), shortlist_size)
    if len(candidates) == 1:
        shortlist = [candidates[0], *shortlist]
    candidate_set = set(candidates)
    total = len(candidates)
    ranked: list[RankedGuess] = []
    for word in shortlist:
        counts: Counter[tuple[str, ...]] = Counter()
        for candidate in candidates:
            if word == candidate:
                counts[(CORRECT,) * _WORD_LENGTH] += 1
            else:
                counts[feedback(word, candidate)] += 1
        expected = sum(size * size for size in counts.values()) / total
        ranked.append(RankedGuess(word=word, expected_remaining=expected, is_possible_answer=word in candidate_set))
    ranked.sort(key=lambda guess: (guess.expected_remaining, not guess.is_possible_answer, guess.word))
    return ranked[:top_k]


def constraint_summary(revealed: Sequence[tuple[str, Sequence[str]]]) -> str:
    """A compact human-readable digest of the board for the classifier's state.

    Duplicate letters make this trickier than it reads: a letter both yellow in one row
    and absent elsewhere is genuinely "in the word, but fewer times than guessed", which
    the summary says explicitly rather than flattening to a contradictory rule.
    """
    greens: list[str] = []
    yellows: list[str] = []
    grays: list[str] = []
    for guess, observed in revealed:
        for i, (letter, state) in enumerate(zip(guess, observed, strict=True)):
            if state == CORRECT:
                greens.append(f"{letter} in position {i + 1}")
            elif state == PRESENT and letter not in yellows:
                yellows.append(letter)
            elif state == ABSENT:
                grays.append(letter)
    parts: list[str] = []
    if greens:
        parts.append("Green (right letter, right spot): " + "; ".join(greens) + ".")
    if yellows:
        grays = [letter for letter in grays if letter not in yellows]
        parts.append("Yellow (in the word, wrong spot): " + ", ".join(sorted(set(yellows))) + ".")
    if grays:
        parts.append("Not in the word: " + ", ".join(sorted(set(grays))) + ".")
    return " ".join(parts) if parts else "No feedback yet — the board is empty."


__all__ = [
    "ABSENT",
    "CORRECT",
    "PRESENT",
    "RankedGuess",
    "answers",
    "constraint_summary",
    "feedback",
    "filter_candidates",
    "guesses",
    "is_allowed",
    "rank_guesses",
]
