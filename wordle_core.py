"""
wordle_core.py — Shared core for the Wordle-solving agents.

Conceptual framing
-------------------
Wordle is a Partially Observable Markov Decision Process (POMDP):

  * The *hidden state* is the secret answer word, which we never observe directly.
  * Each guess produces an *observation*: the 5-slot green/yellow/gray feedback pattern.
  * Our *belief state* is the set of answers still consistent with everything seen
    so far. Because Wordle's feedback is deterministic, the Bayes filter collapses to
    a simple rule: a candidate answer survives iff it would have produced exactly the
    feedback patterns we actually observed for every guess made.

This module provides the three pieces both agents share:

  1. Word-list loading        -> load_words()
  2. The feedback function    -> feedback(guess, answer)   (correct duplicate handling)
  3. The belief filter        -> filter_candidates(candidates, guess, pattern)

The two agents differ ONLY in how they pick the next guess from the belief state;
they reuse everything here.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Pattern encoding
# ---------------------------------------------------------------------------
# Each of the 5 slots gets one of three values:
GRAY = 0    # letter not in the answer (or no unused copy remains)
YELLOW = 1  # letter in the answer but in the wrong slot
GREEN = 2   # letter in the correct slot
#
# A full pattern is a 5-tuple of these values, e.g. (2, 0, 1, 0, 0).
# Tuples are hashable, so they work directly as dict keys when we bucket
# guesses by the pattern they produce. The all-green pattern means
# the guess is correct.

Pattern = Tuple[int, int, int, int, int]

WORD_LEN = 5
ALL_GREEN: Pattern = (GREEN,) * WORD_LEN

# A pattern can also be encoded as a single base-3 integer in [0, 242], since
# 3**5 = 243. This compact form lets the guess x answer lookup table
# fit in a numpy uint8 array, and lets patterns be used as plain array indices
# when bucketing. (GREEN,)*5 encodes to 242.
NUM_PATTERNS = 3 ** WORD_LEN  # 243
ALL_GREEN_ID = NUM_PATTERNS - 1  # 242

# Default on-disk locations for the word lists.
_DATA_DIR = Path(__file__).resolve().parent / "data"
ANSWERS_PATH = _DATA_DIR / "answers.txt"
ALLOWED_GUESSES_PATH = _DATA_DIR / "allowed_guesses.txt"


# ---------------------------------------------------------------------------
# 1. Word-list loading
# ---------------------------------------------------------------------------
def _read_word_file(path: Path) -> List[str]:
    """Read a newline-delimited word file, returning lowercase 5-letter words."""
    with open(path, "r", encoding="utf-8") as f:
        words = [line.strip().lower() for line in f]
    words = [w for w in words if w]  # drop blank lines
    bad = [w for w in words if len(w) != WORD_LEN or not w.isalpha()]
    if bad:
        raise ValueError(f"{path} contains non-5-letter words, e.g. {bad[:5]}")
    return words


def load_words(
    answers_path: Path = ANSWERS_PATH,
    allowed_path: Path = ALLOWED_GUESSES_PATH,
) -> Tuple[List[str], List[str]]:
    """Load the two word lists.

    Returns
    -------
    (answers, allowed_guesses)
        ``answers`` is the ~2,315 possible secret words (the candidate space the
        belief filter tracks). ``allowed_guesses`` is the ~12,972 word superset
        the agents are permitted to guess from. ``answers`` is a subset of
        ``allowed_guesses``.
    """
    answers = _read_word_file(answers_path)
    allowed = _read_word_file(allowed_path)
    return answers, allowed


# ---------------------------------------------------------------------------
# 2. Feedback function (the heart of the whole project)
# ---------------------------------------------------------------------------
def feedback(guess: str, answer: str) -> Pattern:
    """Return Wordle's color pattern for ``guess`` against the secret ``answer``.

    Uses the correct **two-pass** algorithm so duplicate letters are scored the
    way real Wordle scores them:

      Pass 1 (greens): mark every slot where guess[i] == answer[i] GREEN, and
        *consume* that letter from a tally of the answer's remaining letters.
      Pass 2 (yellows): scan the non-green slots left-to-right; a letter is
        YELLOW only while an *unused* copy of it remains in the tally, otherwise
        GRAY. Consuming on use is what stops a guessed letter from getting more
        yellows/greens than there are copies in the answer.

    Example of why the two passes matter — guess "ALLEY" vs answer "LOLLY":
      * Pass 1 greens slot 2 (both 'L') and slot 4 (both 'L'); the answer has
        three L's, two now consumed, one left.
      * Pass 2: the leading 'A' is gray; the first 'L' (slot 1) claims the last
        remaining L as YELLOW; 'E' is gray; 'Y' is green.
      Result: (GRAY, YELLOW, GREEN, GRAY, GREEN) = (0, 1, 2, 0, 2).
    """
    if len(guess) != WORD_LEN or len(answer) != WORD_LEN:
        raise ValueError("feedback() expects two 5-letter words")

    pattern = [GRAY] * WORD_LEN
    # Tally of answer letters still available to match against (greens removed first).
    remaining = Counter(answer)

    # Pass 1: greens. Consume the matched letter so it can't also score a yellow.
    for i in range(WORD_LEN):
        if guess[i] == answer[i]:
            pattern[i] = GREEN
            remaining[guess[i]] -= 1

    # Pass 2: yellows, left-to-right, only while an unused copy remains.
    for i in range(WORD_LEN):
        if pattern[i] == GREEN:
            continue
        letter = guess[i]
        if remaining[letter] > 0:
            pattern[i] = YELLOW
            remaining[letter] -= 1
        # else stays GRAY

    return tuple(pattern)  # type: ignore[return-value]


def encode_pattern(pattern: Pattern) -> int:
    """Encode a 5-slot pattern tuple as a base-3 integer in [0, 242]."""
    pid = 0
    for v in pattern:
        pid = pid * 3 + v
    return pid


def decode_pattern(pid: int) -> Pattern:
    """Inverse of :func:`encode_pattern`."""
    slots = []
    for _ in range(WORD_LEN):
        slots.append(pid % 3)
        pid //= 3
    return tuple(reversed(slots))  # type: ignore[return-value]


def pattern_id(guess: str, answer: str) -> int:
    """feedback() composed with encode_pattern(): the integer pattern directly.

    Kept logically identical to :func:`feedback` (the two-pass algorithm) so the
    lookup table and the tuple-based core can never disagree; the test suite
    asserts ``encode_pattern(feedback(g, a)) == pattern_id(g, a)`` over samples.
    """
    return encode_pattern(feedback(guess, answer))


# ---------------------------------------------------------------------------
# 3. Belief filter (the Bayes-filter update for this POMDP)
# ---------------------------------------------------------------------------
def filter_candidates(
    candidates: Sequence[str],
    guess: str,
    pattern: Pattern,
) -> List[str]:
    """Belief-state update: keep candidates consistent with an observation.

    Given the current belief state (``candidates``), a ``guess`` we made, and the
    ``pattern`` we actually observed, return only those candidate answers that
    *would have produced that exact pattern*. This is the deterministic Bayes
    filter for Wordle — a one-liner because the likelihood of an observation
    given a candidate is 1 if consistent and 0 otherwise.
    """
    return [c for c in candidates if feedback(guess, c) == pattern]


# ---------------------------------------------------------------------------
# Small convenience helpers (handy for the CLI and for debugging)
# ---------------------------------------------------------------------------
def pattern_to_str(pattern: Pattern, style: str = "ascii") -> str:
    """Render a pattern for display.

    ``style="ascii"`` (default) -> 'G'/'Y'/'_', which prints on every console
    (the Windows cp1252 terminal cannot encode the emoji squares).
    ``style="emoji"`` -> '🟩🟨⬜' squares for the write-up; only use it where the
    output stream is UTF-8.
    """
    glyphs = {
        "ascii": {GRAY: "_", YELLOW: "Y", GREEN: "G"},
        "emoji": {GRAY: "⬜", YELLOW: "🟨", GREEN: "🟩"},
    }
    table = glyphs[style]
    return "".join(table[v] for v in pattern)


def is_solved(pattern: Pattern) -> bool:
    """True iff the pattern is all-green (the guess equals the answer)."""
    return pattern == ALL_GREEN


if __name__ == "__main__":
    # Tiny smoke test / demo when run directly.
    answers, allowed = load_words()
    print(f"Loaded {len(answers)} answers, {len(allowed)} allowed guesses.")
    demo = feedback("alley", "lolly")
    print(f"feedback('alley','lolly') = {demo}  {pattern_to_str(demo)}")
