"""
agents.py — The two Wordle agents.

Both agents share the belief-tracking core (wordle_core) and the precomputed
pattern table (pattern_table). They differ ONLY in how they score a candidate
guess against the current belief state:

  * Agent A — InformationGainAgent (expected-utility / average-case):
        For a guess, the remaining answers split into buckets by the pattern each
        would produce. Treat the bucket sizes as a probability distribution over
        observations and compute its Shannon entropy H = -sum(p * log2 p). Pick the
        guess that MAXIMIZES H — i.e. the guess expected to shrink the belief state
        the most on average.

  * Agent B — MinimaxAgent (adversarial / worst-case):
        Same bucketing, but score a guess by its LARGEST bucket (the worst-case
        number of answers still consistent if the adversary chose that pattern).
        Pick the guess whose worst case is SMALLEST.

Shared design notes
-------------------
* Common interface: ``agent.next_guess(candidates, allowed_guesses=None)`` returns
  the next guess word. ``candidates`` is the current belief state (list of still-
  possible answers); ``allowed_guesses`` optionally overrides the guess pool.
* Practical rule (both agents): with only 1-2 candidates left, just guess one of
  them rather than optimizing — you cannot do better than a coin flip at 2, and
  optimizing wastes a guess that could be the answer.
* Opener: the first move is the most expensive to compute and is identical for
  every game, so it is configurable. Pass ``opener="salet"`` (etc.) to use a fixed
  strong opener, or ``opener=None`` to compute it once (the result is memoized so
  the benchmark pays for it a single time, not once per game). Computed, Agent A
  opens SOARE and Agent B opens RAISE.
* Tie-break: a total order, so a tie is never left to list order. Each agent sorts
  by its own metric, then prefers a guess that is still a live candidate (it might
  be the answer and can win outright), then by the *other* agent's metric, then
  alphabetically. The TypeScript port on the portfolio site applies the same order
  and reaches the same guess for every belief state.
* Decisions are memoized per belief state. The agents are deterministic, so two
  different answers that leave the same candidates get the same guess, which is
  what makes the full 2,315-game benchmark finish in about 25 seconds.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from wordle_core import NUM_PATTERNS, ALL_GREEN_ID
from pattern_table import PatternTable


class LookupAgent:
    """Base class: shared belief-aware machinery; subclasses define the score.

    Parameters
    ----------
    table : PatternTable
        The precomputed guess x answer pattern matrix.
    name : str
        Display name for benchmarking output.
    opener : str | None
        Fixed first guess, or None to compute the first guess from the metric.
    guess_pool : {"allowed", "answers"}
        Where to draw candidate guesses from. "allowed" (default) uses the full
        ~12,972-word list (standard, strongest). "answers" restricts guesses to the
        ~2,315 possible answers (faster, slightly weaker) — handy for quick runs.
    """

    def __init__(
        self,
        table: PatternTable,
        name: str,
        opener: Optional[str] = None,
        guess_pool: str = "allowed",
    ):
        self.table = table
        self.name = name
        self.opener = opener.lower() if opener else None
        if guess_pool not in ("allowed", "answers"):
            raise ValueError("guess_pool must be 'allowed' or 'answers'")
        self.guess_pool = guess_pool

        # Default pool of guess words and their row indices into the table.
        pool_words = table.allowed if guess_pool == "allowed" else table.answers
        self._pool_words: List[str] = list(pool_words)
        self._pool_rows: np.ndarray = table.guess_indices(self._pool_words)

        # The full initial belief state size — used to detect "the first move".
        self._n_initial = len(table.answers)
        # Memoized computed opener (only used when self.opener is None).
        self._computed_opener: Optional[str] = None
        self._decisions: dict = {}

        # Lookup table x*log2(x) for integer bucket sizes (with 0->0). Used by the
        # entropy agent to compute H without per-cell float logs:
        #   H = log2(n) - (1/n) * sum_c c*log2(c).
        # A bucket can hold at most every answer, so indices 0..n_initial suffice.
        x = np.arange(self._n_initial + 1, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            self._xlogx = np.where(x > 0, x * np.log2(x), 0.0)

    # -- interface ---------------------------------------------------------
    def next_guess(
        self,
        candidates: Sequence[str],
        allowed_guesses: Optional[Sequence[str]] = None,
    ) -> str:
        """Choose the next guess given the current belief state ``candidates``."""
        if not candidates:
            raise ValueError("next_guess called with no candidates")

        # First move: use the configured/ memoized opener if we have one.
        is_first_move = len(candidates) == self._n_initial
        if is_first_move:
            if self.opener is not None:
                return self.opener
            if self._computed_opener is not None:
                return self._computed_opener

        # Practical rule: 1-2 candidates left -> just guess one of them.
        if len(candidates) <= 2:
            return candidates[0]

        # Determine the guess pool (rows + words) for this call.
        if allowed_guesses is None:
            pool_rows, pool_words = self._pool_rows, self._pool_words
        else:
            pool_words = list(allowed_guesses)
            pool_rows = self.table.guess_indices(pool_words)

        cand_cols = self.table.answer_indices(candidates)
        key = cand_cols.tobytes() if allowed_guesses is None else None
        if key is not None and key in self._decisions:
            return self._decisions[key]
        choice = self._pick(pool_rows, pool_words, cand_cols, set(candidates))
        if key is not None:
            self._decisions[key] = choice

        if is_first_move and self.opener is None:
            self._computed_opener = choice  # memoize the expensive first move
        return choice

    # -- shared scoring ----------------------------------------------------
    def _bucket_counts(self, pool_rows: np.ndarray, cand_cols: np.ndarray) -> np.ndarray:
        """Bucket sizes for every pool guess over the candidate answers.

        Returns an int array of shape ``(len(pool_rows), 243)`` where
        ``counts[g, p]`` is how many candidate answers would give pool-guess ``g``
        the encoded pattern ``p``. This single vectorized step is the heart of both
        agents — entropy reads the whole distribution, minimax reads only the max.

        ``np.ix_`` builds the submatrix directly. Indexing the two axes separately
        (``table[rows][:, cols]``) would copy the entire 30 MB table first, which
        costs about seven times as much per move.
        """
        sub = self.table.table[np.ix_(pool_rows, cand_cols)]   # (G, n) uint8
        g = sub.shape[0]
        # Combine (guess, pattern) into one bin index and count in a single pass.
        flat = (np.arange(g, dtype=np.int64)[:, None] * NUM_PATTERNS + sub).ravel()
        counts = np.bincount(flat, minlength=g * NUM_PATTERNS).reshape(g, NUM_PATTERNS)
        return counts

    def _metrics(self, pool_rows, cand_cols, pool_words, candidate_set):
        """Both agents' scores for every pool guess, from one bucketing pass.

        Returns ``(bits, worst, live)``: expected information gain in bits, the
        largest bucket, and whether each guess is still a live candidate. Each
        agent uses one of the first two as its objective and the other to break
        ties, so computing both costs nothing extra.
        """
        counts = self._bucket_counts(pool_rows, cand_cols)     # (G, 243)
        n = cand_cols.size
        # H = log2(n) - (1/n) * sum_c c*log2(c), via the precomputed x*log2(x)
        # table (an integer gather, far cheaper than 3.15M float logs per move).
        bits = np.log2(n) - self._xlogx[counts].sum(axis=1) / n
        # Two guesses can be mathematically tied on information yet differ in the
        # last bits of the float, because this form and the one the TypeScript
        # port uses round differently. Rounding before the comparison lets the
        # explicit rules below settle those ties instead of the arithmetic.
        bits = np.round(bits, 9)
        # An all-green bucket ends the game rather than leaving candidates behind,
        # so it is not a worst case. It never holds more than one answer anyway.
        counts[:, ALL_GREEN_ID] = 0
        worst = counts.max(axis=1)
        live = np.fromiter((w in candidate_set for w in pool_words),
                           dtype=bool, count=len(pool_words))
        return bits, worst, live

    @staticmethod
    def _first(pool_words, *keys) -> str:
        """Lowest guess under a total order, given lexsort keys (last = primary).

        The pool is alphabetical, so index order is alphabetical order; using it
        as the final key leaves no tie unbroken. That matters for more than
        tidiness — it is what lets the TypeScript port reproduce these results
        word for word.
        """
        return pool_words[np.lexsort(keys)[0]]

    # -- to be implemented by each agent -----------------------------------
    def _pick(self, pool_rows, pool_words, cand_cols, candidate_set) -> str:
        raise NotImplementedError


class InformationGainAgent(LookupAgent):
    """Agent A: maximize expected information gain (Shannon entropy of buckets)."""

    def __init__(self, table, opener=None, guess_pool="allowed", name="InfoGain"):
        super().__init__(table, name, opener=opener, guess_pool=guess_pool)

    def _pick(self, pool_rows, pool_words, cand_cols, candidate_set) -> str:
        bits, worst, live = self._metrics(pool_rows, cand_cols, pool_words, candidate_set)
        # Most bits first; then a guess that could itself win; then the smaller
        # worst case; then alphabetical.
        return self._first(pool_words, worst, ~live, -bits)


class MinimaxAgent(LookupAgent):
    """Agent B: minimize the worst-case bucket (largest consistent group)."""

    def __init__(self, table, opener=None, guess_pool="allowed", name="Minimax"):
        super().__init__(table, name, opener=opener, guess_pool=guess_pool)

    def _pick(self, pool_rows, pool_words, cand_cols, candidate_set) -> str:
        bits, worst, live = self._metrics(pool_rows, cand_cols, pool_words, candidate_set)
        # Smallest worst case first; then a guess that could itself win; then more
        # bits; then alphabetical.
        return self._first(pool_words, -bits, ~live, worst)


def make_agent(kind: str, table: PatternTable, **kwargs) -> LookupAgent:
    """Factory: kind in {'infogain', 'minimax'}."""
    kind = kind.lower()
    if kind in ("infogain", "info", "entropy", "a"):
        return InformationGainAgent(table, **kwargs)
    if kind in ("minimax", "mm", "b"):
        return MinimaxAgent(table, **kwargs)
    raise ValueError(f"unknown agent kind: {kind!r}")
