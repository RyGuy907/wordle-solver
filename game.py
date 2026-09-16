"""
game.py — Play a full Wordle game with an agent driving the belief filter.

This ties the core to an agent: starting from the full answer list as the belief
state, the agent picks a guess, we score it against the hidden answer, update the
belief state with the Bayes filter, and repeat until solved or the 6-guess limit is
hit. The benchmark replays this for every answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from wordle_core import feedback, filter_candidates, is_solved, pattern_to_str
from agents import LookupAgent

MAX_GUESSES = 6


@dataclass
class GameResult:
    """Outcome of one game."""
    answer: str
    guesses: List[str] = field(default_factory=list)
    patterns: List[Tuple[int, ...]] = field(default_factory=list)
    solved: bool = False

    @property
    def num_guesses(self) -> int:
        """Guesses used. For the histogram, an unsolved game counts as a loss."""
        return len(self.guesses)

    def transcript(self, style: str = "ascii") -> str:
        lines = []
        for g, p in zip(self.guesses, self.patterns):
            lines.append(f"  {g.upper()}  {pattern_to_str(p, style)}")
        status = f"solved in {self.num_guesses}" if self.solved else "FAILED (>6)"
        return "\n".join(lines) + f"\n  [{status}] answer={self.answer.upper()}"


def play_game(
    agent: LookupAgent,
    answer: str,
    answers: Sequence[str],
    allowed_guesses: Optional[Sequence[str]] = None,
    max_guesses: int = MAX_GUESSES,
) -> GameResult:
    """Let ``agent`` play a single game against the hidden ``answer``.

    ``answers`` is the initial belief state (full candidate list). The agent's
    guess pool is its own configured pool unless ``allowed_guesses`` is given.
    """
    candidates: List[str] = list(answers)
    result = GameResult(answer=answer)

    for _ in range(max_guesses):
        guess = agent.next_guess(candidates, allowed_guesses)
        pattern = feedback(guess, answer)
        result.guesses.append(guess)
        result.patterns.append(pattern)

        if is_solved(pattern):
            result.solved = True
            break

        # Belief-state update: keep only answers consistent with this observation.
        candidates = filter_candidates(candidates, guess, pattern)
        if not candidates:
            # Should never happen if the true answer is in the initial list.
            break

    return result


if __name__ == "__main__":
    # Quick demo: play a couple of games with each agent and print transcripts.
    from pattern_table import PatternTable
    from agents import InformationGainAgent, MinimaxAgent

    pt = PatternTable.load_or_build()
    answers = pt.answers

    for AgentCls, opener in [(InformationGainAgent, "salet"), (MinimaxAgent, "salet")]:
        agent = AgentCls(pt, opener=opener)
        print(f"\n=== {agent.name} (opener={opener}) ===")
        # All targets below are in the 2,315-answer set (the only words the belief
        # filter tracks). 'mummy'/'jolly' are deliberately tricky double-letter words.
        for target in ["jolly", "vivid", "crane", "mummy"]:
            res = play_game(agent, target, answers)
            print(res.transcript())
