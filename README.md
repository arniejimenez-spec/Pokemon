# Pokémon TCG AI Battle — Agent

My entry for the [Kaggle Pokémon TCG AI Battle Challenge (Simulation)](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle).
A learned policy that plays the Pokémon Trading Card Game on the `cabt` engine: a
priority-driven heuristic bootstraps a behaviorally-cloned network, which is then
fine-tuned by self-play reinforcement learning against a deliberately diverse
opponent pool. See [WRITEUP.md](WRITEUP.md) for the full strategic narrative
(Strategy Category submission) and [EXPERIMENTS.md](EXPERIMENTS.md) for the
complete, dated experiment log with every ladder result.

> **Note:** the competition engine and the Pokémon card data are **not** included in this
> repository. They are proprietary (the engine is not open-source; card names/text are
> Pokémon/Nintendo IP) and must be obtained from the Kaggle competition Data page. See
> [Setup](#setup).

## Status

**Standing submission: `models/policy_rl_pool_it12.npz`** — a behaviorally-cloned,
self-play-RL-tuned policy network, ladder-confirmed at **535.3**, **+58.3 points
concurrent** over the prior best. Active experimentation is currently paused (see
[Where this stands](#where-this-stands)); the codebase is left in a state ready to
resume.

## Approach

The project went through three architectures, in order, each replaced because the
ladder — not local testing — showed the prior one didn't work as evaluated:

1. **Rule-based heuristic** (`agents/heuristic.py`) — a priority-driven policy
   covering every selection context the engine can present (bench, attach, evolve,
   abilities, max-damage/KO attacks, discards). Beats a random baseline ~10–0.
   Still used as: the BC teacher, the RL rollout policy for opponents, and the
   safety-net fallback if the network ever produces a non-finite output.
   **Ladder: 601** (settled baseline).
2. **Determinized-rollout search + opponent modeling** (`agents/search_agent.py`,
   `agents/search_core.py`, `agents/opponent_model.py`) — an anytime searcher that
   samples plausible hidden-information worlds and rolls candidate moves forward
   with the heuristic driving both sides. Won 67–75% locally. **Ladder: 456 — an
   active regression.** The local win was circular: the search's opponent-rollout
   policy *was* the heuristic it was being measured against. Kept as research code,
   not shipped. See WRITEUP.md §3 for the full analysis.
3. **Learned policy: behavioral cloning + self-play RL** (`agents/policy_agent.py`,
   `train/gen_bc_data.py`, `train/train_bc.py`, `train/rl_loop.py`) — **the current
   shipped agent.** The action space is variable-length (a different set of legal
   options every decision, no fixed action space), so the network scores each
   offered option independently — `logit = MLP([state, option])` — and softmaxes
   over whatever the engine actually offers. Inference is pure numpy (no torch
   dependency in the submission bundle), completing in <10ms against a
   600s/episode budget.

### Feature versions

| Version | What it encodes | Status |
|---|---|---|
| v1 (`agents/features.py`) | Board state + option attributes (counts, types, HP) | Shipped in earlier submissions |
| v2 (`agents/features_v2.py`) | v1 + a learned per-option card-identity embedding | **Currently shipped** |
| v3 (`agents/features_v3.py`) | v2 + four state-level "id-bags" (own hand / deck-remaining / own discard / opponent discard), each a multiset of card ids summed through a learned embedding table | Built, verified (numpy/torch parity 1.5e-06), paused on a training-budget confound — not disproven |

### Deck

`decks/mega_lucario.py` — Mega Lucario ex, chosen deliberately for **pilotability**:
one evolution step (not Stage 2), one energy color (Fighting), one clear attacker
line with a cheap/finisher attack pair. Two real-world decklists were later ported
and tested to validate this reasoning — see [Deck research](#deck-research).

## The central methodological finding: evaluation circularity

Every regression this project shipped came from measuring an agent against an
opponent that shared an assumption with the agent itself — the search's rollout
policy for the opponent, or a later policy's own mirror match against its
predecessor. Every confirmed gain came from measuring against opponents and
conditions the agent had no assumptions about. This is documented in depth in
[WRITEUP.md](WRITEUP.md) and is the reason the project now uses:

- **A diverse RL training pool** (`train/rl_loop.py`) — frozen past-champion
  policies + the scripted heuristic + multiple deck archetypes, not primarily
  self-copies.
- **A diverse, held-out evaluation panel** (`eval_panel.py`) — candidates are
  scored against 5 distinct opponents; one archetype (Latias, Psychic — also our
  own weakness type) is *never* in the training pool, serving as a genuine
  generalization test. Shipping requires beating the reigning champion in a
  **direct head-to-head**, not just improving an aggregate score.

## Ladder results (ground truth)

All comparisons are **concurrent** — absolute ratings drift by 100+ points across
a single day as the field strengthens, so only same-moment comparisons mean
anything. See [EXPERIMENTS.md](EXPERIMENTS.md) for the full row-by-row log.

| Milestone | Ladder rating | Verdict |
|---|---|---|
| Heuristic | 601 | Baseline |
| Search + opponent model | 456 | Regression — circular local eval |
| BC + self-play RL (v1) | +105 vs. concurrent heuristic | First confirmed win |
| Mirror-selected v2 identity-embedding model | ≈ tie (6.6 pts) | Same circularity, different disguise |
| **Anti-mirror pivot (current)** | **535.3, +58.3 vs. concurrent prior best** | **Largest confirmed gain — current submission** |

## Where this stands

Four further single-variable experiments were run against the current champion to
see whether the anti-mirror recipe compounds: more RL iterations (null), a wider
opponent pool (null), a lower learning rate (broke a repeated overfitting pattern,
promising but inconclusive), and the v3 richer-feature rebuild (paused on a
training-budget confound). Two deck-scouting attempts (a real Dragapult ex/Dusknoir
list, and a real Mega Lucario list) validated the deck-simplicity reasoning above
but found no upgrade over the current build. Given the competition deadline, active
experimentation is paused and the current champion is preserved as the standing
submission — see EXPERIMENTS.md's closing entries for the full reasoning and the
concrete next steps if resumed.

## Deck research

`decks/dragapult_dusknoir.py` and `decks/mega_lucario_v2.py` are real tournament
decklists ported from competitive play data, used to stress-test the "simple deck
= more pilotable" hypothesis:

- **Dragapult ex/Dusknoir** — legal in our card pool, but the heuristic failed to
  pilot it: a second evolution line never completed in 15/15 test games and 53%
  of games ended by deck-out instead of combat.
- **Real-world Mega Lucario** — fully pilotable (90% vs. a random baseline, both
  evolution lines completing reliably), but lost to our own simpler build
  head-to-head (40%).

Both results affirm that deck *simplicity* is itself a strategic parameter for an
automated pilot, independent of a deck's strength in expert human hands.

## Repo layout

```
agents/              heuristic (BC teacher/RL rollout/fallback), policy_agent.py (shipped
                     inference), features.py/_v2.py/_v3.py (encoders), search_agent.py +
                     search_core.py + opponent_model.py (research, not shipped), random_agent.py
decks/               mega_lucario.py (shipped deck), gauntlet.py (eval/training opponents),
                     dragapult_dusknoir.py + mega_lucario_v2.py (deck-research candidates)
train/               gen_bc_data.py (self-play data gen), train_bc.py (BC trainer, --fv 1/2/3),
                     rl_loop.py (self-play RL fine-tuning)
harness.py           seat-swapped match runner for two agents
eval_panel.py        current selection tool: diverse panel incl. a training-holdout deck
make_submission.py   builds submission.tar.gz for any agent, self-validates the bundle
log_submission.py    appends/updates rows in EXPERIMENTS.md from a built bundle's manifest
smoke_test.py        random-vs-random engine sanity check
EXPERIMENTS.md        the full, dated ladder-result log with reasoning for every row
WRITEUP.md           Strategy Category submission draft
select_ckpt*.py,      one-off checkpoint-selection scripts from specific historical cycles;
run_gauntlet*.py,     see EXPERIMENTS.md for the cycle each belongs to. eval_panel.py is the
diag_*.py            current tool -- these predate it.
```

## Setup

1. Download the simulator (SDK) and card data from the competition **Data** page.
2. Place the engine package at `cg/` in the repo root (so `from cg.api import ...` resolves).
3. `python smoke_test.py` to confirm the engine loads.
4. `python harness.py` to run a match between two agents; `python eval_panel.py <model.npz> --games 250` to score a candidate against the diverse panel.

Requires Python 3.x + numpy. Training additionally needs torch (see below); the
*submission* never does — inference is pure numpy.

## Submitting

The agents are a **portfolio, not branches** — they share the encoder, harness, decks
and engine wrapper, so they live side by side on `main`. One command builds any of them:

```bash
python make_submission.py --agent policy --model models/policy_rl_pool_it12.npz --note "anti-mirror pivot"
python make_submission.py --agent heuristic --note "baseline"
```

Each build assembles the bundle from scratch, writes a **`MANIFEST.json`** inside it
(agent, git SHA, model sha256, build time) so a downloaded tarball always says what it
is, and **validates itself** — extract, import, play a real self-play game, and assert
the model actually loaded (a silent heuristic fallback would otherwise pass while
shipping the wrong agent).

Then record it, and fill in the rating once the ladder settles:

```bash
python log_submission.py --note "anti-mirror pivot"       # appends a row to EXPERIMENTS.md
python log_submission.py --row 8 --rating 535.3 --verdict "+58.3 concurrent — largest confirmed gain"
```

Every submitted build is also git-tagged (`sub-<date>-<agent>`), so the exact code
behind any ladder score is recoverable. See [EXPERIMENTS.md](EXPERIMENTS.md) — this
exists because we once could not tell whether a rating belonged to the heuristic or
the search agent, and lost a day to the ambiguity.

## Training pipeline

```bash
# 1. Generate self-play data with the heuristic (--fv 1/2/3 selects the feature version)
python train/gen_bc_data.py --games 6000 --workers 7 --fv 2 --out data/

# 2. Upload the .npz as a private Kaggle Dataset, train on a GPU notebook (T4 -- NOT
#    P100: sm_60 is incompatible with Kaggle's torch build), download the weights
python train/train_bc.py --data data/bc_data_v2.npz --fv 2 --out models/policy_bc_v2.npz

# 3. Self-play RL fine-tuning -- runs locally on CPU (the model is small; self-play
#    itself is CPU-bound, so a GPU round-trip costs more than it saves)
python train/rl_loop.py --model models/policy_bc_v2.npz --iters 16 --games 1400 --workers 7

# 4. Select: panel-evaluate candidates against the current champion, ship only on a
#    clear head-to-head win that also holds the training-holdout deck
python eval_panel.py ckpt/policy_it12.npz --champion models/policy_rl_pool_it12.npz --games 250
```
