"""
test_wordle_core.py — Unit tests for the shared core.

Run with either:
    python -m pytest test_wordle_core.py -v      (if pytest is installed)
    python test_wordle_core.py                   (falls back to a bare runner)

The duplicate-letter cases are the important ones: they are exactly where a
naive single-pass feedback function goes wrong.
"""

from wordle_core import (
    GRAY, YELLOW, GREEN, ALL_GREEN, ALL_GREEN_ID, NUM_PATTERNS,
    feedback, filter_candidates, is_solved, load_words,
    encode_pattern, decode_pattern, pattern_id,
)


# ---------------------------------------------------------------------------
# Basic, non-duplicate cases
# ---------------------------------------------------------------------------
def test_exact_match_all_green():
    assert feedback("crane", "crane") == (GREEN, GREEN, GREEN, GREEN, GREEN)
    assert feedback("crane", "crane") == ALL_GREEN
    assert is_solved(feedback("crane", "crane"))


def test_all_gray_no_overlap():
    # No shared letters at all.
    assert feedback("fizzy", "crwth") == (GRAY, GRAY, GRAY, GRAY, GRAY)


def test_simple_mixed_no_duplicates():
    # answer "pause": p? in guess "speak"
    #   s -> in answer, wrong slot  -> YELLOW
    #   p -> in answer, wrong slot  -> YELLOW
    #   e -> in answer, wrong slot  -> YELLOW
    #   a -> in answer, wrong slot  -> YELLOW (answer has 'a' at slot 1, guess slot 3)
    #   k -> not in answer          -> GRAY
    assert feedback("speak", "pause") == (YELLOW, YELLOW, YELLOW, YELLOW, GRAY)


def test_green_takes_priority_over_yellow_position():
    # guess "alarm" vs answer "aroma": both start with 'a' -> green slot 0.
    g = feedback("alarm", "aroma")
    assert g[0] == GREEN


# ---------------------------------------------------------------------------
# Duplicate-letter cases — the heart of the spec
# ---------------------------------------------------------------------------
def test_alley_vs_lolly():
    # The canonical example from the project brief.
    #   A : not... actually 'a' not in LOLLY      -> GRAY
    #   L : LOLLY has three L's; two consumed by greens (slots 2,4),
    #       one left -> this L claims it           -> YELLOW
    #   L : matches answer slot 2 ('l')            -> GREEN
    #   E : not in answer                          -> GRAY
    #   Y : matches answer slot 4 ('y')            -> GREEN
    assert feedback("alley", "lolly") == (GRAY, YELLOW, GREEN, GRAY, GREEN)


def test_guess_has_two_of_a_letter_answer_has_one():
    # "llama" vs "child": the answer holds a single 'l', at slot 3.
    #   l (slot0): answer has one 'l' (slot3, not consumed) -> YELLOW
    #   l (slot1): no more 'l' copies left                  -> GRAY
    #   a (slot2): not in "child"                            -> GRAY
    #   m (slot3): not in "child"                            -> GRAY
    #   a (slot4): not in "child"                            -> GRAY
    assert feedback("llama", "child") == (YELLOW, GRAY, GRAY, GRAY, GRAY)


def test_two_yellows_when_answer_has_two_copies():
    # "madam" holds two 'm' and two 'a', so a guess with three 'm' can collect
    # at most two of them. Guessing "mamma":
    #   m (slot0): matches 'm' at answer slot0 -> GREEN, consume one m (1 m left)
    #   a (slot1): matches 'a' at answer slot1 -> GREEN, consume one a (1 a left)
    #   m (slot2): answer slot2 is 'd'; 1 m left -> YELLOW, consume it (0 m left)
    #   m (slot3): no m left                     -> GRAY
    #   a (slot4): answer slot4 is 'm'; 1 a left -> YELLOW, consume it
    assert feedback("mamma", "madam") == (GREEN, GREEN, YELLOW, GRAY, YELLOW)


def test_green_consumes_before_yellow_assignment():
    # The ordering case: greens claim their letters before any yellow is handed
    # out, so a guess can show fewer yellows than it has copies of the letter.
    # answer "level" = l e v e l; guess "eevee" = e e v e e.
    #   Greens first: slot1 e, slot2 v, slot3 e. That consumes both of the
    #     answer's 'e' and its 'v'.
    #   Yellows next: slot0 'e' and slot4 'e' find no unused copy -> GRAY.
    # Scoring yellows first would wrongly light up slot0.
    assert feedback("eevee", "level") == (GRAY, GREEN, GREEN, GREEN, GRAY)


def test_duplicate_guess_letter_one_green_one_gray():
    # guess "abbey" vs answer "above": one 'b' in answer (slot2).
    #   a (0): matches 'a'            -> GREEN
    #   b (1): answer slot1 'b'?  above = a b o v e -> slot1 IS 'b' -> GREEN
    #   b (2): no 'b' left            -> GRAY
    #   e (3): answer has 'e' at slot4, available -> YELLOW
    #   y (4): not in answer          -> GRAY
    assert feedback("abbey", "above") == (GREEN, GREEN, GRAY, YELLOW, GRAY)


# ---------------------------------------------------------------------------
# Belief filter
# ---------------------------------------------------------------------------
def test_filter_keeps_only_consistent():
    candidates = ["crane", "crate", "slate", "trace", "lolly"]
    guess = "crane"
    pattern = feedback(guess, "crate")  # observation produced by the true answer
    survivors = filter_candidates(candidates, guess, pattern)
    # "crate" must survive; "lolly" (totally different) must not.
    assert "crate" in survivors
    assert "lolly" not in survivors
    # Every survivor must reproduce the observed pattern by construction.
    assert all(feedback(guess, c) == pattern for c in survivors)


def test_filter_self_consistency_on_full_list():
    # For a random-ish answer, filtering the full answer list by the true
    # pattern must always retain the true answer.
    answers, _ = load_words()
    answer = "vivid"
    guess = "video"
    pattern = feedback(guess, answer)
    survivors = filter_candidates(answers, guess, pattern)
    assert answer in survivors


def test_feedback_is_not_symmetric():
    # feedback() takes its arguments in a fixed order, and swapping them is not
    # the same question. Duplicate letters show it most clearly: "lolly" holds
    # three 'l' and "alley" only two, so the two directions disagree.
    assert feedback("alley", "lolly") == (GRAY, YELLOW, GREEN, GRAY, GREEN)
    assert feedback("lolly", "alley") == (YELLOW, GRAY, GREEN, GRAY, GREEN)


# ---------------------------------------------------------------------------
# Word-list sanity
# ---------------------------------------------------------------------------
def test_word_lists_load_and_sizes():
    answers, allowed = load_words()
    assert len(answers) == 2315
    assert len(allowed) == 12972
    assert set(answers) <= set(allowed)   # answers are a subset of allowed
    assert all(len(w) == 5 for w in answers)
    assert all(len(w) == 5 for w in allowed)


# ---------------------------------------------------------------------------
# Pattern encoding (used by the lookup table)
# ---------------------------------------------------------------------------
def test_encode_decode_roundtrip():
    # Every one of the 243 patterns must survive encode -> decode.
    seen = set()
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    for e in range(3):
                        pat = (a, b, c, d, e)
                        pid = encode_pattern(pat)
                        seen.add(pid)
                        assert 0 <= pid < NUM_PATTERNS
                        assert decode_pattern(pid) == pat
    assert len(seen) == NUM_PATTERNS  # all 243 distinct
    assert encode_pattern(ALL_GREEN) == ALL_GREEN_ID


def test_pattern_id_matches_feedback():
    # pattern_id must equal encode_pattern(feedback(...)) for tricky words too.
    for g, a in [("alley", "lolly"), ("mamma", "madam"), ("eevee", "level"),
                 ("crane", "crane"), ("abbey", "above"), ("llama", "child")]:
        assert pattern_id(g, a) == encode_pattern(feedback(g, a))


# ---------------------------------------------------------------------------
# Bare-bones fallback runner (so `python test_wordle_core.py` works w/o pytest)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
