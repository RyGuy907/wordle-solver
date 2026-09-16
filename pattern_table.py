"""
pattern_table.py — Precomputed guess x answer feedback lookup table.

Why this exists
---------------
Scoring a single guess for the information-gain or minimax agent means computing
the feedback pattern of that guess against every remaining answer. At the very
first move that is ~12,972 guesses x ~2,315 answers ~= 30 million pattern
evaluations, and the benchmark replays a full game for all 2,315 answers. Calling
the pure-Python :func:`feedback` 30M+ times per move is far too slow.

The fix is to compute every (guess, answer) pattern ONCE, store it as a compact
``uint8`` matrix (each pattern encoded to a base-3 int in [0, 242]), and cache it
to disk. After that, an agent's per-move work becomes pure numpy array slicing
and ``bincount`` — milliseconds instead of seconds.

Layout
------
``table`` has shape ``(n_allowed, n_answers)`` and dtype ``uint8``:
    table[g, a] == pattern_id(allowed[g], answers[a])
Row = guess index, column = answer index. ~12,972 x 2,315 ~= 30 MB in memory.

The table is keyed to the exact word lists it was built from; the cache filename
encodes the list sizes and a content hash so a stale cache is never silently used.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from wordle_core import WORD_LEN, ALL_GREEN_ID, load_words

_CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Base-3 slot weights for encoding a (n, 5) pattern array to (n,) pattern ids.
# encode_pattern() folds left-to-right as pid = pid*3 + v, i.e. slot 0 is the
# most significant trit -> weights [81, 27, 9, 3, 1].
_SLOT_WEIGHTS = np.array([81, 27, 9, 3, 1], dtype=np.uint16)


def _words_to_codes(words: Sequence[str]) -> np.ndarray:
    """Map words to an (n, 5) int8 array of letter codes (a->0 .. z->25)."""
    arr = np.frombuffer("".join(words).encode("ascii"), dtype=np.uint8)
    return (arr.reshape(len(words), WORD_LEN) - ord("a")).astype(np.int8)


def _letter_counts(codes: np.ndarray) -> np.ndarray:
    """Per-word letter histogram: (n, 26) int8, counts[w, c] = #occurrences."""
    n = codes.shape[0]
    counts = np.zeros((n, 26), dtype=np.int8)
    rows = np.repeat(np.arange(n), WORD_LEN)
    np.add.at(counts, (rows, codes.ravel()), 1)
    return counts


def _row_patterns(
    guess_codes: np.ndarray,      # (5,)  letter codes of one guess
    ans_codes: np.ndarray,        # (n_a, 5)
    ans_counts: np.ndarray,       # (n_a, 26)
) -> np.ndarray:
    """Vectorized two-pass feedback of one guess against ALL answers at once.

    Returns a (n_a,) uint8 array of encoded pattern ids. This is exactly the
    algorithm in wordle_core.feedback (greens consume first, then yellows
    left-to-right while an unused copy remains), applied across every answer
    simultaneously with numpy. The test suite checks it cell-for-cell against the
    scalar pattern_id so the speed-up cannot introduce a discrepancy.
    """
    n = ans_codes.shape[0]
    pat = np.zeros((n, WORD_LEN), dtype=np.uint8)

    # Pass 1: greens, and consume the matched letter from each answer's tally.
    green = ans_codes == guess_codes[None, :]      # (n, 5) bool
    pat[green] = 2
    avail = ans_counts.copy()                       # (n, 26)
    for i in range(WORD_LEN):
        gi = guess_codes[i]
        avail[green[:, i], gi] -= 1

    # Pass 2: yellows left-to-right, only where an unused copy remains.
    for i in range(WORD_LEN):
        gi = guess_codes[i]
        can = (~green[:, i]) & (avail[:, gi] > 0)
        pat[can, i] = 1
        avail[can, gi] -= 1

    return (pat.astype(np.uint16) * _SLOT_WEIGHTS[None, :]).sum(axis=1).astype(np.uint8)


def _list_hash(answers: Sequence[str], allowed: Sequence[str]) -> str:
    """Short content hash of the two word lists, to validate a cached table."""
    h = hashlib.sha1()
    h.update(("\n".join(answers)).encode())
    h.update(b"||")
    h.update(("\n".join(allowed)).encode())
    return h.hexdigest()[:12]


class PatternTable:
    """The guess x answer pattern matrix plus word<->index lookups.

    Attributes
    ----------
    answers, allowed : list[str]
        The word lists, in the row/column order used by ``table``.
    table : np.ndarray  (uint8, shape [n_allowed, n_answers])
        ``table[g, a]`` is the encoded pattern of ``allowed[g]`` vs ``answers[a]``.
    ans_index, allowed_index : dict[str, int]
        word -> index maps for columns (answers) and rows (allowed guesses).
    """

    def __init__(self, answers: List[str], allowed: List[str], table: np.ndarray):
        self.answers = answers
        self.allowed = allowed
        self.table = table
        self.ans_index: Dict[str, int] = {w: i for i, w in enumerate(answers)}
        self.allowed_index: Dict[str, int] = {w: i for i, w in enumerate(allowed)}

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, answers: List[str], allowed: List[str], verbose: bool = True) -> "PatternTable":
        """Compute the full table from scratch (no cache)."""
        n_g, n_a = len(allowed), len(answers)
        table = np.empty((n_g, n_a), dtype=np.uint8)
        # Precompute the answer-side arrays once; each guess reuses them.
        ans_codes = _words_to_codes(answers)
        ans_counts = _letter_counts(ans_codes)
        guess_codes_all = _words_to_codes(allowed)
        t0 = time.perf_counter()
        for g in range(n_g):
            table[g] = _row_patterns(guess_codes_all[g], ans_codes, ans_counts)
            if verbose and (g + 1) % 2000 == 0:
                elapsed = time.perf_counter() - t0
                rate = (g + 1) / elapsed
                eta = (n_g - g - 1) / rate
                print(f"  built {g + 1:>6}/{n_g} guess rows "
                      f"({elapsed:5.1f}s elapsed, ~{eta:5.1f}s left)")
        if verbose:
            print(f"  table built in {time.perf_counter() - t0:.1f}s")
        return cls(answers, allowed, table)

    # -- caching -----------------------------------------------------------
    @classmethod
    def load_or_build(
        cls,
        answers: List[str] | None = None,
        allowed: List[str] | None = None,
        cache_dir: Path = _CACHE_DIR,
        verbose: bool = True,
    ) -> "PatternTable":
        """Return the table, loading the on-disk cache if it matches the lists."""
        if answers is None or allowed is None:
            answers, allowed = load_words()
        cache_dir.mkdir(exist_ok=True)
        key = _list_hash(answers, allowed)
        path = cache_dir / f"patterns_{len(allowed)}x{len(answers)}_{key}.npy"
        if path.exists():
            if verbose:
                print(f"Loading cached pattern table: {path.name}")
            table = np.load(path)
            return cls(answers, allowed, table)
        if verbose:
            print(f"No cache found; building pattern table "
                  f"({len(allowed)} x {len(answers)})...")
        obj = cls.build(answers, allowed, verbose=verbose)
        np.save(path, obj.table)
        if verbose:
            print(f"Cached pattern table -> {path.name}")
        return obj

    # -- index helpers -----------------------------------------------------
    def answer_indices(self, words: Sequence[str]) -> np.ndarray:
        """Map a list of answer words to a numpy array of column indices."""
        return np.fromiter((self.ans_index[w] for w in words), dtype=np.int32,
                           count=len(words))

    def guess_indices(self, words: Sequence[str]) -> np.ndarray:
        """Map a list of guess words to a numpy array of row indices."""
        return np.fromiter((self.allowed_index[w] for w in words), dtype=np.int32,
                           count=len(words))


if __name__ == "__main__":
    # Build (or load) and report a couple of sanity facts.
    pt = PatternTable.load_or_build()
    print(f"table shape {pt.table.shape}, dtype {pt.table.dtype}, "
          f"{pt.table.nbytes / 1e6:.1f} MB")
    # The diagonal-ish sanity check: a word vs itself must be all-green.
    if "crane" in pt.allowed_index and "crane" in pt.ans_index:
        g = pt.allowed_index["crane"]
        a = pt.ans_index["crane"]
        assert pt.table[g, a] == ALL_GREEN_ID
        print("self-vs-self all-green check: OK")
