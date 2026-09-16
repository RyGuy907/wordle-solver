"""
solve.py — Watch the agents solve specific words you choose.

Examples
--------
    python solve.py crane mummy vivid          # both agents, default opener
    python solve.py --agent infogain salet     # just Agent A
    python solve.py --opener crane mummy        # use a different opener
    python solve.py --opener computed mummy     # compute each agent's first move
    python solve.py --ascii mummy               # force ASCII (no ANSI colors)

Notes
-----
* The agents track only the 2,315-answer list as candidates, so a target that is
  not an official answer (only in the larger allowed-guess list) is unsolvable by
  design — solve.py warns you when that happens.
* Any 5-letter word in the allowed list is fine as the --opener.
"""

from __future__ import annotations

import argparse

from pattern_table import PatternTable
from agents import make_agent
from game import play_game
from human_play import render_row, _enable_windows_ansi


def main() -> None:
    ap = argparse.ArgumentParser(description="Solve specific words with the agents.")
    ap.add_argument("words", nargs="+", help="one or more 5-letter target words")
    ap.add_argument("--agent", default="both",
                    choices=["both", "infogain", "minimax"])
    ap.add_argument("--opener", default="salet",
                    help="fixed opener word, or 'computed' to derive each agent's first move")
    ap.add_argument("--pool", default="allowed", choices=["allowed", "answers"])
    ap.add_argument("--ascii", action="store_true", help="disable ANSI colors")
    args = ap.parse_args()

    color = not args.ascii
    if color:
        _enable_windows_ansi()

    table = PatternTable.load_or_build(verbose=False)
    answers = table.answers
    allowed_set = set(table.allowed)
    answer_set = set(answers)

    # Validate the requested targets up front.
    targets = [w.strip().lower() for w in args.words]
    for w in targets:
        if w not in allowed_set:
            print(f"! {w.upper()} is not a valid 5-letter word in the allowed list "
                  f"(it cannot appear as a real Wordle answer).")
        elif w not in answer_set:
            print(f"! {w.upper()} is in the allowed-guess list but NOT the 2,315 "
                  f"answers - the agents track only answers, so it is unsolvable "
                  f"by design (it will run out of candidates).")

    opener = None if args.opener == "computed" else args.opener
    kinds = ["infogain", "minimax"] if args.agent == "both" else [args.agent]

    for kind in kinds:
        agent = make_agent(kind, table, opener=opener, guess_pool=args.pool)
        shown_opener = args.opener if opener else "computed"
        print(f"\n=== {agent.name} (opener={shown_opener}, pool={args.pool}) ===")
        for target in targets:
            if target not in answer_set:
                # Still allow a manual run, but the belief filter may empty out.
                print(f"\n[{target.upper()}] skipped (not an answer-list word)")
                continue
            res = play_game(agent, target, answers)
            print()
            for g, p in zip(res.guesses, res.patterns):
                print("  " + render_row(g, p, color))
            status = f"solved in {res.num_guesses}" if res.solved else "FAILED (>6)"
            print(f"  [{status}]")


if __name__ == "__main__":
    main()
