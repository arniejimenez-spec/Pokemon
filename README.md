# Pokémon TCG AI Battle — Agent

My entry for the [Kaggle Pokémon TCG AI Battle Challenge (Simulation)](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle).
An agent that plays the Pokémon Trading Card Game on the `cabt` engine, combining a
hand-written heuristic policy with a determinized-rollout search and opponent modeling.

> **Note:** the competition engine and the Pokémon card data are **not** included in this
> repository. They are proprietary (the engine is not open-source; card names/text are
> Pokémon/Nintendo IP) and must be obtained from the Kaggle competition Data page. See
> [Setup](#setup).

## Status

**The submitted agent is the plain heuristic** (`agents/heuristic.py`). The search agent
and opponent model are kept in the repo as research code, but they are **not shipped** —
they scored *worse* than the heuristic on the live ladder. See
[Ladder results](#ladder-results-ground-truth).

## Approach

- **Heuristic policy** (`agents/heuristic.py`) — a priority-driven policy covering every
  selection context (bench, attach, evolve, abilities, max-damage/KO attacks, discards).
  Beats a random agent ~10–0. Currently the shipped agent; also the rollout policy inside
  the search.
- **Determinized-rollout search** (`agents/search_agent.py`, `agents/search_core.py`) —
  an anytime searcher that fires only at genuine decision points. It samples plausible
  hidden-information worlds, plays each candidate move forward a few turns with the
  heuristic driving both sides, scores the resulting board, and averages over sampled
  worlds within a wall-clock budget. *Research code — not currently shipped.*
- **Opponent modeling** (`agents/opponent_model.py`) — infers the opponent's dominant
  energy type from visible Pokémon/energy and builds a prior for the hidden bulk from the
  opponent's actually-observed cards. *Research code — not currently shipped.*

## Ladder results (ground truth)

| Submission | Kaggle skill rating |
|---|---|
| Heuristic only | **601.2** |
| Determinized search + opponent model | **456.4** |

Ratings start at μ₀=600, so the search agent was *actively losing*. Kaggle agent logs ruled
out timeouts and crashes (zero stderr; ~4.6s used of a 600s per-episode budget) — the
losses were pure gameplay.

### Why the local numbers were wrong

Local evaluation said the opposite (search beat the heuristic 67–75%, opponent modeling
"added" ~8 points). Those results were **circular**:

1. The search uses `HeuristicPolicy` as its **opponent-rollout policy**, and every local
   test used that same heuristic as the actual opponent — so the search's model of its
   opponent was correct *by construction*. Real ladder agents don't play like it.
2. The v1 opponent-model type templates were built from the **same attackers and shell as
   the gauntlet decks they were tested against** — the model had the answer key.
3. At 150ms/decision with ~13ms rollouts, the search got only **~2 rollouts per
   candidate** — noise rather than signal on a stochastic game.

**Lesson:** a local win measured against the same policy the agent internally assumes
proves nothing. The ladder is the only ground truth.

## Layout

```
agents/            heuristic (shipped), search core/agent + opponent model (research), random baseline
decks/             deck lists (mega_lucario.py = main deck; gauntlet.py = eval opponents)
harness.py         seat-swapped match runner for two agents
run_gauntlet.py    3-way A/B: heuristic vs mirror-search vs modeled-search
run_gauntlet2.py   A/B for the evidence-adapted (v2) opponent prior
diag_headtohead.py modeled-vs-mirror head-to-head diagnostic
diag_hardened.py   quality check after the search timing hardening
make_submission.py builds submission.tar.gz (main.py + deck.csv + modules at top level)
smoke_test.py      random-vs-random engine sanity check
submission/        main.py + deck.csv (Kaggle entrypoint)
```

> Caveat: `run_gauntlet*.py` measure against `HeuristicPolicy`, which is exactly the
> circularity described above. Treat their output as a smoke test, not evidence.

## Setup

1. Download the simulator (SDK) and card data from the competition **Data** page.
2. Place the engine package at `cg/` in the repo root (so `from cg.api import ...` resolves).
3. `python smoke_test.py` to confirm the engine loads.
4. `python harness.py` or `python run_gauntlet.py` to evaluate.
5. `python make_submission.py` to build the upload bundle.

Requires Python 3.x. No third-party dependencies beyond the competition SDK.
