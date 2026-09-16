"""
benchmark.py — Run each agent against every answer and compare.

For each agent we replay a full game (capped at 6 guesses) against all ~2,315
possible answers and collect:

  * average guesses (over all games; an unsolved game is counted as its 6 used
    guesses), worst case, and % solved within 6
  * the full histogram of guess counts: how many answers were solved in 1..6
    (plus a "fail" bucket for >6)
  * the per-answer guess count, so human_play.py can compare a human against the
    agents on the EXACT same target words.

Results are cached to ``results/benchmark_<key>.json`` keyed by agent config, so a
re-run loads instantly instead of recomputing. Pass ``--force`` to recompute.

Usage
-----
    python benchmark.py                         # both agents, default openers
    python benchmark.py --opener computed       # compute each first move instead
    python benchmark.py --opener crane          # fixed opener for both
    python benchmark.py --agents infogain       # just one agent
    python benchmark.py --pool answers          # guess only from answer set (fast)
    python benchmark.py --limit 200             # quick smoke run on first 200
    python benchmark.py --force                 # ignore cache, recompute
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from pattern_table import PatternTable
from agents import make_agent, LookupAgent
from game import play_game, MAX_GUESSES

_RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Default fixed openers per agent (configurable from the CLI). These are strong,
# well-known starting words so "fixed opener" runs are reproducible.
DEFAULT_OPENERS = {"infogain": "salet", "minimax": "salet"}


@dataclass
class BenchmarkResult:
    """Aggregated results for one agent over the full answer set."""
    label: str                       # display name, e.g. "InfoGain (salet)"
    agent_kind: str
    opener: str                      # "computed" or the fixed opener word
    guess_pool: str
    histogram: Dict[int, int] = field(default_factory=dict)   # guesses -> count
    fails: int = 0                                            # games needing >6
    per_answer: Dict[str, int] = field(default_factory=dict)  # answer -> guesses
    total_games: int = 0
    seconds: float = 0.0

    # -- derived stats -----------------------------------------------------
    @property
    def solved(self) -> int:
        return sum(self.histogram.values())

    @property
    def solve_pct(self) -> float:
        return 100.0 * self.solved / self.total_games if self.total_games else 0.0

    @property
    def avg_guesses(self) -> float:
        """Mean guesses over all games (a failed game contributes its 6 guesses)."""
        total = sum(k * v for k, v in self.histogram.items()) + self.fails * MAX_GUESSES
        return total / self.total_games if self.total_games else 0.0

    @property
    def worst_case(self) -> int:
        # The histogram is pre-seeded with a zero for every bucket 1..6, so the
        # count has to be checked; taking max() over the keys alone reports 6 for
        # every agent regardless of how it actually did.
        solved_worst = max((k for k, v in self.histogram.items() if v), default=0)
        return MAX_GUESSES + 1 if self.fails else solved_worst

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "agent_kind": self.agent_kind,
            "opener": self.opener,
            "guess_pool": self.guess_pool,
            "histogram": {str(k): v for k, v in self.histogram.items()},
            "fails": self.fails,
            "per_answer": self.per_answer,
            "total_games": self.total_games,
            "seconds": self.seconds,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkResult":
        r = cls(
            label=d["label"], agent_kind=d["agent_kind"], opener=d["opener"],
            guess_pool=d["guess_pool"], fails=d["fails"],
            per_answer=d.get("per_answer", {}),
            total_games=d["total_games"], seconds=d.get("seconds", 0.0),
        )
        r.histogram = {int(k): v for k, v in d["histogram"].items()}
        return r


def _result_path(agent_kind: str, opener: str, pool: str, n: int) -> Path:
    return _RESULTS_DIR / f"benchmark_{agent_kind}_{opener}_{pool}_{n}.json"


def run_agent_benchmark(
    agent: LookupAgent,
    answers: List[str],
    opener_label: str,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """Replay ``agent`` against every answer (or the first ``limit``)."""
    targets = answers[:limit] if limit else answers
    result = BenchmarkResult(
        label=f"{agent.name} ({opener_label})",
        agent_kind=agent.name.lower(),
        opener=opener_label,
        guess_pool=agent.guess_pool,
        total_games=len(targets),
    )
    result.histogram = {k: 0 for k in range(1, MAX_GUESSES + 1)}

    t0 = time.perf_counter()
    for i, target in enumerate(targets):
        game = play_game(agent, target, answers)
        if game.solved:
            result.histogram[game.num_guesses] += 1
            result.per_answer[target] = game.num_guesses
        else:
            result.fails += 1
            result.per_answer[target] = MAX_GUESSES + 1   # sentinel for "did not solve"
        if verbose and (i + 1) % 250 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{agent.name}] {i + 1}/{len(targets)} games "
                  f"({elapsed:5.1f}s, running avg "
                  f"{sum(k*v for k,v in result.histogram.items())/max(1,sum(result.histogram.values())):.3f})")
    result.seconds = time.perf_counter() - t0
    return result


def get_or_run(
    agent_kind: str,
    table: PatternTable,
    opener: str,
    pool: str,
    limit: Optional[int],
    force: bool,
) -> BenchmarkResult:
    """Load a cached benchmark if present and matching, else run and cache it."""
    _RESULTS_DIR.mkdir(exist_ok=True)
    n = limit if limit else len(table.answers)
    path = _result_path(agent_kind, opener, pool, n)
    if path.exists() and not force:
        print(f"Loading cached benchmark: {path.name}")
        return BenchmarkResult.from_dict(json.loads(path.read_text()))

    opener_arg = None if opener == "computed" else opener
    agent = make_agent(agent_kind, table, opener=opener_arg, guess_pool=pool)
    print(f"Running {agent.name} benchmark (opener={opener}, pool={pool}, "
          f"games={n})...")
    result = run_agent_benchmark(agent, table.answers, opener, limit=limit)
    path.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"  saved -> {path.name} ({result.seconds:.1f}s)")
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary_table(results: List[BenchmarkResult]) -> None:
    """Print a side-by-side summary table plus the guess-count histograms."""
    print("\n" + "=" * 64)
    print("BENCHMARK SUMMARY")
    print("=" * 64)
    header = f"{'Agent':<22}{'Avg':>7}{'Worst':>7}{'Solve%':>9}{'Fails':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.label:<22}{r.avg_guesses:>7.3f}{r.worst_case:>7}"
              f"{r.solve_pct:>8.1f}%{r.fails:>7}")

    print("\nGuess-count distribution (count of answers solved in N guesses):")
    cols = "".join(f"{k:>7}" for k in range(1, MAX_GUESSES + 1)) + f"{'>6':>7}"
    print(f"{'Agent':<22}{cols}")
    print("-" * (22 + 7 * (MAX_GUESSES + 1)))
    for r in results:
        cells = "".join(f"{r.histogram.get(k, 0):>7}" for k in range(1, MAX_GUESSES + 1))
        print(f"{r.label:<22}{cells}{r.fails:>7}")
    print("=" * 64)


def plot_comparison(results: List[BenchmarkResult], out_path: Path) -> None:
    """Grouped bar chart of each agent's guess-count distribution."""
    import matplotlib
    matplotlib.use("Agg")  # headless: write a file, never open a window
    import matplotlib.pyplot as plt
    import numpy as np

    buckets = list(range(1, MAX_GUESSES + 1)) + [MAX_GUESSES + 1]  # last = ">6"
    labels = [str(k) for k in range(1, MAX_GUESSES + 1)] + [">6"]
    x = np.arange(len(buckets))
    width = 0.8 / max(1, len(results))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, r in enumerate(results):
        heights = [r.histogram.get(k, 0) for k in range(1, MAX_GUESSES + 1)] + [r.fails]
        ax.bar(x + i * width, heights, width, label=f"{r.label}  (avg {r.avg_guesses:.2f})")

    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Guesses to solve")
    ax.set_ylabel("Number of answers")
    ax.set_title("Wordle agents: guess-count distribution over all answers")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved comparison chart -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark the Wordle agents.")
    ap.add_argument("--agents", default="infogain,minimax",
                    help="comma list: infogain,minimax")
    ap.add_argument("--opener", default=None,
                    help="'computed', or a fixed opener word (default: per-agent SALET)")
    ap.add_argument("--pool", default="allowed", choices=["allowed", "answers"])
    ap.add_argument("--limit", type=int, default=None, help="run only first N answers")
    ap.add_argument("--force", action="store_true", help="ignore cache, recompute")
    ap.add_argument("--no-chart", action="store_true")
    args = ap.parse_args()

    table = PatternTable.load_or_build()
    kinds = [k.strip() for k in args.agents.split(",") if k.strip()]

    results = []
    for kind in kinds:
        opener = args.opener or DEFAULT_OPENERS.get(kind, "salet")
        results.append(get_or_run(kind, table, opener, args.pool, args.limit, args.force))

    print_summary_table(results)
    if not args.no_chart:
        _RESULTS_DIR.mkdir(exist_ok=True)
        plot_comparison(results, _RESULTS_DIR / "comparison_chart.png")


if __name__ == "__main__":
    main()
