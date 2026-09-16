"""
assist.py — Let an agent solve a real Wordle (e.g. today's NYT puzzle).

You are the eyes: the agent suggests a guess, you type it into the real Wordle,
and then you tell the agent the color pattern you got back. The agent runs the
same belief filter as the benchmark and suggests the next guess, until solved.

Pattern input
-------------
Type 5 characters, one per tile, using any of these:
    green  -> g  or 2
    yellow -> y  or 1
    gray   -> x  or b or _ or . or 0
e.g. for green/gray/yellow/gray/green you'd type:  gxyxg

Commands at the prompt
----------------------
    use WORD   the agent's suggestion wasn't accepted / you played a different
               word — substitute WORD as the guess, then enter its pattern
    undo       step back one turn (e.g. you typed the wrong pattern)
    quit       stop

Usage
-----
    python assist.py                       # InfoGain agent, SALET opener
    python assist.py --agent minimax       # use the minimax agent
    python assist.py --opener crane        # different opener
    python assist.py --opener computed     # let the agent compute its first move
    python assist.py --ascii               # no ANSI colors
"""

from __future__ import annotations

import argparse

from wordle_core import feedback, filter_candidates, is_solved, ALL_GREEN, GRAY, YELLOW, GREEN
from pattern_table import PatternTable
from agents import make_agent
from human_play import render_row, _enable_windows_ansi
from game import MAX_GUESSES

# Map every accepted pattern character to a slot value.
_CHAR_TO_VAL = {
    "g": GREEN, "2": GREEN,
    "y": YELLOW, "1": YELLOW,
    "x": GRAY, "b": GRAY, "_": GRAY, ".": GRAY, "-": GRAY, "0": GRAY,
}


def parse_pattern(text: str):
    """Parse a 5-char feedback string into a pattern tuple, or None if invalid."""
    text = text.strip().lower().replace(" ", "")
    if len(text) != 5:
        return None
    try:
        return tuple(_CHAR_TO_VAL[ch] for ch in text)
    except KeyError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Solve a real Wordle with an agent.")
    ap.add_argument("--agent", default="infogain", choices=["infogain", "minimax"])
    ap.add_argument("--opener", default="salet",
                    help="fixed opener word, or 'computed' to derive the first move")
    ap.add_argument("--pool", default="allowed", choices=["allowed", "answers"])
    ap.add_argument("--ascii", action="store_true", help="disable ANSI colors")
    args = ap.parse_args()

    color = not args.ascii
    if color:
        _enable_windows_ansi()

    table = PatternTable.load_or_build(verbose=False)
    answers = table.answers

    opener = None if args.opener == "computed" else args.opener
    agent = make_agent(args.agent, table, opener=opener, guess_pool=args.pool)

    print(f"Wordle assistant - {agent.name} agent")
    print("Type the color pattern after each guess (e.g. 'gxyxg').")
    print("Commands: 'use WORD', 'undo', 'quit'.\n")

    # history holds candidate lists so 'undo' can step back a turn.
    candidates = list(answers)
    history = [candidates]
    turn = 1

    while turn <= MAX_GUESSES:
        if not candidates:
            print("\n! No candidate answers remain. Either a pattern was entered")
            print("  wrong, or today's answer isn't in the tracked answer list.")
            print("  Type 'undo' to fix the last pattern, or 'quit'.")
            cmd = input("> ").strip().lower()
            if cmd == "undo" and len(history) > 1:
                history.pop()
                candidates = history[-1]
                turn -= 1
                continue
            return

        suggestion = agent.next_guess(candidates, None)
        print(f"--- Guess {turn}/{MAX_GUESSES} ---")
        print(f"  Suggested guess: {suggestion.upper()}   "
              f"({len(candidates)} possible answer(s) left)")
        if len(candidates) <= 8:
            print(f"  Remaining: {', '.join(c.upper() for c in candidates)}")

        guess = suggestion
        # Read the pattern (or a command) for this guess.
        while True:
            raw = input(f"  Pattern for {guess.upper()} (or use/undo/quit): ").strip()
            low = raw.lower()
            if low == "quit":
                print("Stopped.")
                return
            if low == "undo":
                if len(history) > 1:
                    history.pop()
                    candidates = history[-1]
                    turn -= 1
                    print("  (stepped back one turn)\n")
                else:
                    print("  nothing to undo")
                break
            if low.startswith("use "):
                word = low.split(None, 1)[1].strip()
                if len(word) == 5 and word.isalpha():
                    guess = word
                    print(f"  using your guess: {guess.upper()}")
                else:
                    print("  ! 'use' needs a 5-letter word")
                continue
            pattern = parse_pattern(raw)
            if pattern is None:
                print("  ! enter 5 chars from g/y/x (e.g. gxyxg)")
                continue

            # Valid pattern: show it, update belief, advance.
            print("  " + render_row(guess, pattern, color))
            if is_solved(pattern):
                print(f"\nSolved in {turn} guess(es)! The word is {guess.upper()}.")
                return
            candidates = filter_candidates(candidates, guess, pattern)
            history.append(candidates)
            turn += 1
            print()
            break

    print("Out of guesses (6). The agent couldn't solve it within the limit.")


if __name__ == "__main__":
    main()
