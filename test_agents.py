"""
test_agents.py — Tests for the lookup table and the two agents.

test_wordle_core.py covers the readable, scalar core. The checks here cover the
two places where the project trades readability for speed, and where a silent
disagreement would be hard to spot:

  1. The vectorized pattern table must agree with the scalar feedback function
     cell for cell. Every agent decision reads the table instead of calling
     feedback(), so a single wrong cell would quietly change results.
  2. The agents must be deterministic and must pick the documented openers, since
     the write-up and the TypeScript port both quote those words.

Run with either:
    python -m pytest test_agents.py -v
    python test_agents.py
"""

import random

from wordle_core import feedback, filter_candidates, is_solved, pattern_id
from pattern_table import PatternTable
from agents import InformationGainAgent, MinimaxAgent
from game import play_game

# One shared table for the whole module; building it is the expensive part.
_TABLE = PatternTable.load_or_build(verbose=False)


# ---------------------------------------------------------------------------
# The lookup table against the scalar implementation
# ---------------------------------------------------------------------------
def test_table_matches_scalar_feedback_on_sample():
    # A fixed seed keeps the sample reproducible; 400 guess rows against every
    # answer is about 900,000 cells, which is enough to catch a systematic error
    # in the vectorized two-pass build without checking all 30 million.
    rng = random.Random(1234)
    rows = rng.sample(range(len(_TABLE.allowed)), 400)
    for g in rows:
        guess = _TABLE.allowed[g]
        for a, answer in enumerate(_TABLE.answers):
            assert _TABLE.table[g, a] == pattern_id(guess, answer), (guess, answer)


def test_table_covers_duplicate_letter_words():
    # The sample above is random, so pin the duplicate-letter cases explicitly.
    for guess in ("alley", "mamma", "eevee", "abbey", "llama", "vivid"):
        if guess not in _TABLE.allowed_index:
            continue
        g = _TABLE.allowed_index[guess]
        for a, answer in enumerate(_TABLE.answers):
            assert _TABLE.table[g, a] == pattern_id(guess, answer), (guess, answer)


def test_self_vs_self_is_all_green():
    from wordle_core import ALL_GREEN_ID
    for word in _TABLE.answers[:200]:
        g = _TABLE.allowed_index[word]
        a = _TABLE.ans_index[word]
        assert _TABLE.table[g, a] == ALL_GREEN_ID


# ---------------------------------------------------------------------------
# Agent behaviour
# ---------------------------------------------------------------------------
def test_computed_openers_are_the_documented_words():
    # These two words are quoted in the README and hardcoded in the TypeScript
    # port, so a change in scoring or tie-breaking should fail here loudly.
    # RAISE is one of five guesses tied at a worst case of 168; it wins the tie
    # because it is a possible answer and carries the most information of those.
    infogain = InformationGainAgent(_TABLE, opener=None)
    minimax = MinimaxAgent(_TABLE, opener=None)
    assert infogain.next_guess(_TABLE.answers) == "soare"
    assert minimax.next_guess(_TABLE.answers) == "raise"


def test_agents_are_deterministic():
    # Same belief state, same guess, every time — the benchmark's decision cache
    # and the port's reproducibility both depend on it.
    candidates = filter_candidates(_TABLE.answers, "soare", feedback("soare", "abled"))
    for AgentCls in (InformationGainAgent, MinimaxAgent):
        agent = AgentCls(_TABLE, opener="soare")
        picks = {agent.next_guess(candidates) for _ in range(3)}
        assert len(picks) == 1


def test_tie_break_prefers_a_live_candidate():
    # After SOARE and CANAL only four answers remain, and several guesses split
    # them perfectly. The agent must take one of the four, because a guess that
    # cannot be the answer throws away a chance to win outright.
    candidates = filter_candidates(_TABLE.answers, "soare", feedback("soare", "abled"))
    candidates = filter_candidates(candidates, "canal", feedback("canal", "abled"))
    assert candidates == ["abled", "alley", "delta", "fella"]
    for AgentCls in (InformationGainAgent, MinimaxAgent):
        agent = AgentCls(_TABLE, opener="soare")
        assert agent.next_guess(candidates) in candidates


def test_two_candidates_shortcut_guesses_a_candidate():
    agent = InformationGainAgent(_TABLE, opener="soare")
    assert agent.next_guess(["abled", "alley"]) == "abled"


def test_agents_solve_a_sample_of_answers():
    # A seeded sample rather than all 2,315 — the full sweep is benchmark.py.
    rng = random.Random(7)
    targets = rng.sample(_TABLE.answers, 25)
    for AgentCls, opener in ((InformationGainAgent, "soare"), (MinimaxAgent, "raise")):
        agent = AgentCls(_TABLE, opener=opener)
        for target in targets:
            result = play_game(agent, target, _TABLE.answers)
            assert result.solved, (AgentCls.__name__, target)
            assert is_solved(result.patterns[-1])
            assert result.guesses[-1] == target


def test_true_answer_always_survives_the_belief_filter():
    # The filter may never discard the real answer, whatever the guess.
    rng = random.Random(99)
    for answer in rng.sample(_TABLE.answers, 40):
        candidates = list(_TABLE.answers)
        for guess in rng.sample(_TABLE.allowed, 4):
            candidates = filter_candidates(candidates, guess, feedback(guess, answer))
            assert answer in candidates


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
