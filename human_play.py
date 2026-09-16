"""
human_play.py — Human-play CLI, logged to the same stats as the agents.

A human plays Wordle in the terminal against either random answers or a FIXED
SEEDED list of answers. Because the agents store a per-answer guess count for
every word, seeding the human's targets lets us compare human vs Agent A vs
Agent B on the EXACT same words.

Flow
----
    python human_play.py --seed 42 --count 5      # play 5 seeded words
    python human_play.py --random --count 3       # play 3 random words
    python human_play.py --seed 42 --count 5 --compare-only   # just show the table
    python human_play.py ... --ascii              # no ANSI colors

After play, results are saved to ``results/human_<seed>_<count>.json`` in the same
format as the agent benchmarks, and a comparison table is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

from wordle_core import feedback, is_solved, GRAY, YELLOW, GREEN
from game import MAX_GUESSES
from benchmark import BenchmarkResult, _RESULTS_DIR, _result_path, DEFAULT_OPENERS

# ---------------------------------------------------------------------------
# Colored rendering
# ---------------------------------------------------------------------------
_ANSI = {
    GREEN: "\033[30;42m",   # black text on green
    YELLOW: "\033[30;43m",  # black text on yellow
    GRAY: "\033[97;100m",   # white text on bright-black (gray)
}
_RESET = "\033[0m"


def _enable_windows_ansi() -> None:
    """Best-effort enable of ANSI escape processing on Windows consoles."""
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # 7 = PROCESSED_OUTPUT | WRAP_AT_EOL | VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:  # noqa: BLE001 — purely cosmetic; fall back to ascii
            pass


def render_row(guess: str, pattern, color: bool) -> str:
    """Render one guessed word with its feedback, colored or ASCII."""
    if color:
        cells = [f"{_ANSI[v]} {ch.upper()} {_RESET}" for ch, v in zip(guess, pattern)]
        return "".join(cells)
    glyph = {GRAY: "_", YELLOW: "Y", GREEN: "G"}
    letters = " ".join(ch.upper() for ch in guess)
    marks = "  ".join(glyph[v] for v in pattern)
    return f"{letters}\n  {marks}"


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------
def seeded_targets(answers: List[str], seed: int, count: int) -> List[str]:
    """Deterministic list of ``count`` answer words for a given ``seed``.

    Sorting first makes the selection independent of the file's load order, so
    the same seed always yields the same words for both human and agents.
    """
    rng = random.Random(seed)
    return rng.sample(sorted(answers), count)


# ---------------------------------------------------------------------------
# Interactive game
# ---------------------------------------------------------------------------
def play_human_game(
    answer: str,
    allowed_set: set,
    color: bool,
    game_no: int,
    total: int,
) -> Optional[int]:
    """Play one interactive game.

    Returns the number of guesses used if solved, ``MAX_GUESSES + 1`` if the word
    was not solved within six, or ``None`` if the player quit the whole session.
    """
    print(f"\n--- Word {game_no}/{total} --- (type 'quit' to stop, 'giveup' to reveal)")
    for turn in range(1, MAX_GUESSES + 1):
        while True:
            raw = input(f"Guess {turn}/{MAX_GUESSES}: ").strip().lower()
            if raw == "quit":
                return None
            if raw == "giveup":
                print(f"  The word was: {answer.upper()}")
                return MAX_GUESSES + 1
            if len(raw) != 5 or not raw.isalpha():
                print("  ! enter a 5-letter word")
                continue
            if raw not in allowed_set:
                print("  ! not in the allowed word list")
                continue
            break

        pattern = feedback(raw, answer)
        print("  " + render_row(raw, pattern, color))
        if is_solved(pattern):
            print(f"  Solved in {turn}!")
            return turn

    print(f"  Out of guesses - the word was: {answer.upper()}")
    return MAX_GUESSES + 1


def run_session(
    targets: List[str],
    allowed_set: set,
    color: bool,
    label: str,
) -> BenchmarkResult:
    """Play through ``targets`` and return results in the agent stats format."""
    result = BenchmarkResult(
        label=label, agent_kind="human", opener="human", guess_pool="-",
        total_games=0,
    )
    result.histogram = {k: 0 for k in range(1, MAX_GUESSES + 1)}

    for i, answer in enumerate(targets, start=1):
        outcome = play_human_game(answer, allowed_set, color, i, len(targets))
        if outcome is None:
            print("\nSession ended early.")
            break
        result.total_games += 1
        if outcome <= MAX_GUESSES:
            result.histogram[outcome] += 1
            result.per_answer[answer] = outcome
        else:
            result.fails += 1
            result.per_answer[answer] = MAX_GUESSES + 1
    return result


# ---------------------------------------------------------------------------
# Comparison against the agents on the SAME words
# ---------------------------------------------------------------------------
def load_agent_per_answer(agent_kind: str, n_answers: int) -> Optional[BenchmarkResult]:
    """Load a full-run agent benchmark (default opener) for comparison, if cached."""
    opener = DEFAULT_OPENERS.get(agent_kind, "salet")
    path = _result_path(agent_kind, opener, "allowed", n_answers)
    if path.exists():
        return BenchmarkResult.from_dict(json.loads(path.read_text()))
    return None


def _avg_over(words: List[str], per_answer: Dict[str, int]) -> Optional[float]:
    vals = [per_answer[w] for w in words if w in per_answer]
    return sum(vals) / len(vals) if vals else None


def print_comparison(
    targets: List[str],
    human: BenchmarkResult,
    agents: Dict[str, BenchmarkResult],
) -> None:
    """Per-word and average comparison of human vs each agent on the SAME words."""
    print("\n" + "=" * 60)
    print("HUMAN vs AGENTS - identical target words")
    print("=" * 60)

    names = ["Human"] + list(agents.keys())
    header = f"{'Word':<8}" + "".join(f"{n:>10}" for n in names)
    print(header)
    print("-" * len(header))

    def fmt(v: Optional[int]) -> str:
        if v is None:
            return "  -"
        return "X" if v > MAX_GUESSES else str(v)

    for w in targets:
        cells = [fmt(human.per_answer.get(w))]
        for k in agents:
            cells.append(fmt(agents[k].per_answer.get(w)))
        print(f"{w.upper():<8}" + "".join(f"{c:>10}" for c in cells))

    print("-" * len(header))
    avg_cells = [_avg_over(targets, human.per_answer)]
    for k in agents:
        avg_cells.append(_avg_over(targets, agents[k].per_answer))
    print(f"{'AVG':<8}" + "".join(
        (f"{a:>10.2f}" if a is not None else f"{'-':>10}") for a in avg_cells))
    print("=" * 60)
    print("(X = not solved within 6;  '-' = no data for that word)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Play Wordle and compare to the agents.")
    ap.add_argument("--seed", type=int, default=42, help="seed for the target list")
    ap.add_argument("--random", action="store_true",
                    help="use a fresh random (unseeded) target list instead")
    ap.add_argument("--count", type=int, default=5, help="number of words to play")
    ap.add_argument("--ascii", action="store_true", help="disable ANSI colors")
    ap.add_argument("--compare-only", action="store_true",
                    help="skip play; just show the saved human result vs agents")
    args = ap.parse_args()

    from pattern_table import PatternTable  # local import: avoids cost on --help
    table = PatternTable.load_or_build(verbose=False)
    answers, allowed = table.answers, table.allowed
    allowed_set = set(allowed)

    color = not args.ascii
    if color:
        _enable_windows_ansi()

    if args.random:
        targets = random.sample(sorted(answers), args.count)
        seed_label = "random"
    else:
        targets = seeded_targets(answers, args.seed, args.count)
        seed_label = str(args.seed)

    human_path = _RESULTS_DIR / f"human_{seed_label}_{args.count}.json"

    if args.compare_only:
        if not human_path.exists():
            print(f"No saved human result at {human_path.name}; play first.")
            return
        human = BenchmarkResult.from_dict(json.loads(human_path.read_text()))
        targets = list(human.per_answer.keys())
    else:
        print("Wordle - human play")
        print(f"Targets: {len(targets)} word(s), seed={seed_label}")
        human = run_session(targets, allowed_set, color, label=f"Human (seed {seed_label})")
        _RESULTS_DIR.mkdir(exist_ok=True)
        human_path.write_text(json.dumps(human.to_dict(), indent=2))
        print(f"\nSaved your results -> {human_path.name}")

    # Compare against agents on the same words (if their benchmarks are cached).
    agents: Dict[str, BenchmarkResult] = {}
    for kind, disp in [("infogain", "InfoGain"), ("minimax", "Minimax")]:
        res = load_agent_per_answer(kind, len(answers))
        if res is not None:
            agents[disp] = res
    if agents:
        print_comparison(targets, human, agents)
    else:
        print("\n(No cached agent benchmarks found - run `python benchmark.py` "
              "to enable the human-vs-agent comparison.)")


if __name__ == "__main__":
    main()
