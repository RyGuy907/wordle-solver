# Wordle-Solving AI Agents

For my AI Algorithms final project I built two agents that solve Wordle. They share a
belief-tracking core and differ only in how they choose the next word, which makes
them a clean comparison; one optimizes the average case and the other optimizes the
worst case. Both solve all 2,315 possible answers without ever failing, and the
project benchmarks them against each other and against my own play on identical
words.

A written walkthrough with a live in-browser version of these agents is at
[ryanjrusson.com/work/wordle](https://ryanjrusson.com/work/wordle).

## Wordle as a POMDP

Wordle is a partially observable Markov decision process. The answer is a hidden
state I never see directly, each color pattern is an observation, and what I
actually track is a belief over which answers are still possible.

| POMDP concept | Wordle |
|---|---|
| Hidden state | the secret answer, never observed directly |
| Action | a guess, drawn from the 12,972-word allowed list |
| Observation | the five-slot green/yellow/gray feedback |
| Belief state | the answers still consistent with everything seen so far |
| Observation model | deterministic: a guess produces exactly one pattern per answer |

Because the feedback is deterministic and the answer never changes, the general
Bayes filter collapses into a much simpler rule. The likelihood of an observation
is 1 if a candidate would have produced it and 0 otherwise, so the update is just a
filter over the candidate list. That is the whole of `filter_candidates`, and both
agents sit on top of it.

## The two agents

Both agents score a guess the same way. The remaining answers are bucketed by the
pattern each would return, which gives up to 243 buckets. A guess that splits the
answers into many small buckets is informative, because whichever bucket comes back
leaves few candidates behind. The agents disagree only about how to read those
bucket sizes.

**Agent A, information gain** (`InformationGainAgent`) treats the bucket sizes as a
distribution and takes the Shannon entropy `H = -Σ p·log2 p`, picking the guess that
maximizes it. Under a uniform prior that entropy is exactly the expected information
gain in bits, so this agent optimizes the average case.

**Agent B, minimax** (`MinimaxAgent`) looks only at each guess's largest bucket and
picks the guess whose largest bucket is smallest. It treats the answer as an
adversary choosing the worst pattern it can, so it optimizes the worst case rather
than the mean.

Both agents break ties the same way, and the order is deterministic all the way
down: the agent's own metric first, then a guess that is still a live candidate and
could win outright, then the other agent's metric, then alphabetical. The last key
matters more than it looks. Without it the winner depends on word-list order, and
the TypeScript port would quietly disagree with this project.

## The opening guess

Every game starts from the same belief state, so each agent's first move is the same
every time and is worth reporting on its own.

| Agent | Opener | Expected bits | Worst-case bucket | Possible answer? |
|---|---|---|---|---|
| Information gain | `soare` | 5.886 | 183 | no |
| Minimax | `raise` | 5.878 | 168 | yes |

`soare` is not in the answer list at all, so the entropy agent spends its first turn
purely on information. Minimax has a genuine tie to resolve; five words reach the
best worst case of 168, and two of them (`arise` and `raise`) are possible answers.
The tie-break takes `raise`, which carries the most information of the five and can
also win outright.

## Results

Every run below covers all 2,315 answers with the full 12,972-word guess pool, and
neither agent ever fails to solve within six.

Each agent playing its own computed opener:

| Agent | Avg | Worst | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|---|
| Information gain (`soare`) | 3.464 | 6 | 0 | 44 | 1,217 | 990 | 63 | 1 |
| Minimax (`raise`) | 3.521 | 5 | 1 | 67 | 1,045 | 1,129 | 73 | 0 |

The trade-off comes out about as the theory predicts. Information gain is better on
average by roughly 0.06 guesses, and minimax never needs a sixth guess while
information gain needs one exactly once, on `waver`. That word sits in a family of
answers differing by a single letter — `wafer`, `wager`, `water`, `waver` — and the
entropy agent walks into a two-candidate coin flip on the fifth turn and loses it.
Avoiding exactly that situation is what the minimax agent gives up average
performance to buy.

Both agents on a shared `salet` opener, for comparison:

| Agent | Avg | Worst | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|---|
| Information gain (`salet`) | 3.433 | 6 | 0 | 79 | 1,205 | 983 | 46 | 2 |
| Minimax (`salet`) | 3.482 | 5 | 0 | 89 | 1,071 | 1,105 | 50 | 0 |

The same ordering holds, which suggests the trade-off is a property of the two
objectives and not an artifact of one opening word. The more interesting result is
that `salet` beats both agents' own openers, by 0.031 for information gain and 0.039
for minimax, even though it scores worse on both agents' criteria at move one; it is
worth 5.835 bits against `soare`'s 5.886, and its worst-case bucket is 221 against
`raise`'s 168. Each agent picks the opener that looks best one move ahead, and one
move ahead is not the same thing as best overall. That is the clearest limitation of
greedy scoring that this project measures, though confirming it properly would take
a search over openers rather than the two data points here.

## Reproducing

```bash
pip install -r requirements.txt

python test_wordle_core.py      # feedback function and belief filter
python test_agents.py           # lookup table vs scalar, agent behaviour

python benchmark.py                    # both agents, shared SALET opener
python benchmark.py --opener computed  # each agent computes its own first move
python benchmark.py --limit 200        # quick run on the first 200 answers
```

The first run builds the 12,972 × 2,315 pattern table, which takes about five
seconds, and caches it to `cache/`. Benchmarks cache to `results/` keyed by agent
configuration, so a second run of the same configuration loads instead of
recomputing; pass `--force` to recompute anyway. A full benchmark of both agents
takes under a minute.

Two more entry points are useful by hand. `solve.py` watches the agents play words
you name, and `assist.py` suggests guesses for a real Wordle you are playing
yourself, taking the colors back from you each turn.

```bash
python solve.py crane mummy vivid
python assist.py --agent minimax
python human_play.py --seed 42 --count 5   # play, then compare against both agents
```

`human_play.py` deals a seeded list of answers so the same words can be replayed
through both agents afterward, which is the only fair way to compare. Session
results are written to `results/human_*.json` and are deliberately not committed.

## Files

| File | Purpose |
|---|---|
| `wordle_core.py` | feedback function with two-pass duplicate handling, belief filter, pattern encoding |
| `pattern_table.py` | precomputed guess × answer pattern table, vectorized and cached |
| `agents.py` | both agents over a shared scoring and tie-breaking base |
| `game.py` | plays one full game with an agent |
| `benchmark.py` | runs an agent over every answer; summary table and chart |
| `solve.py`, `assist.py`, `human_play.py` | interactive entry points |
| `test_wordle_core.py`, `test_agents.py` | tests |
| `data/` | `answers.txt` (2,315), `allowed_guesses.txt` (12,972) |

Word lists come from the cfreshman gists of the original Wordle lists. The answer
list is a subset of the allowed list.

## Performance notes

Scoring one move at the start of a game means evaluating about 12,972 guesses
against 2,315 answers, and the benchmark replays a full game for every answer. Three
things make that tractable. The full pattern table is computed once as a `uint8`
matrix of encoded patterns and cached, which turns each move into array slicing.
Entropy is computed as `log2(n) - (1/n)·Σ c·log2(c)` against a precomputed table of
`x·log2(x)`, which replaces roughly 3.15 million float logarithms per move with an
integer gather. Additionally, the agents memoize their decision per belief state,
since the same state always produces the same guess.

The submatrix is taken with `np.ix_` rather than by indexing the two axes
separately. Indexing them separately copies the whole 30 MB table before selecting
columns, and dropping that copy took the full entropy benchmark from about 65
minutes down to under a minute.

## Limitations

Both agents are greedy and look exactly one move ahead. Neither searches the game
tree, and neither optimizes the thing actually being measured, which is the expected
number of guesses to finish. The `salet` result above is direct evidence that this
costs something.

Also, the belief state tracks only the 2,315 answers rather than all 12,972 allowed
words, and it assumes those answers are uniformly likely. A word outside the answer
list is unsolvable by design, and the interactive tools warn when one is given.
Minimax minimizes the worst-case number of surviving candidates rather than the
worst-case number of remaining guesses, which is a related but not identical
quantity.

If I were to extend this, I would add a second ply of lookahead and compare it
against the greedy version on the same words, and I would try weighting the belief
by word frequency instead of treating all 2,315 answers as equally likely.

## The TypeScript port

The in-browser demo on my portfolio is a separate TypeScript implementation of these
agents running in a Web Worker. It uses the same feedback rule, the same belief
filter, the same bucket scoring and the same tie-break order, and it returns the
same guess count as this project for all 2,315 answers. It skips the pattern table
entirely, because once the opening guess is a committed constant every later turn
faces a small enough candidate set to score directly.

Overall this was a genuinely fun project to build, and the most useful part was that
holding one core fixed and changing only the decision rule made the difference
between the two strategies easy to see.

## License

The code here is MIT licensed; see `LICENSE`. That covers what I wrote, and not the
two word lists in `data/`, which come from the cfreshman gists of the original
Wordle lists and are included only so the benchmarks reproduce. Wordle itself
belongs to the New York Times.

This started as a graded final project. Reading it, running it and building on it
are all fine by me, but if you are taking a similar course, please read it rather
than submit it.
