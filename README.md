# Pokémon TCG AI Battle — Agent

My entry for the [Kaggle Pokémon TCG AI Battle Challenge (Simulation)](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle).
An agent that plays the Pokémon Trading Card Game on the `cabt` engine, combining a
hand-written heuristic policy with a determinized-rollout search and opponent modeling.

> **Note:** the competition engine and the Pokémon card data are **not** included in this
> repository. They are proprietary (the engine is not open-source; card names/text are
> Pokémon/Nintendo IP) and must be obtained from the Kaggle competition Data page. See
> [Setup](#setup).

## Approach

- **Heuristic policy** (`agents/heuristic.py`) — a priority-driven policy covering every
  selection context (bench, attach, evolve, abilities, max-damage/KO attacks, discards).
  Beats a random agent ~10–0. Doubles as the rollout policy inside the search.
- **Determinized-rollout search** (`agents/search_agent.py`, `agents/search_core.py`) —
  an anytime searcher that fires only at genuine decision points. It samples plausible
  hidden-information worlds, plays each candidate move forward a few turns with the
  heuristic driving both sides, scores the resulting board, and averages over sampled
  worlds within a wall-clock budget.
- **Opponent modeling** (`agents/opponent_model.py`) — infers the opponent's dominant
  energy type from their visible Pokémon and attached energy, then uses a type-matched
  deck template as the determinization prior instead of assuming a mirror match.

### Local evaluation (30 games/cell, win rate as the Mega Lucario deck)

| Opponent | Heuristic | Search (mirror prior) | Search (opponent model) |
|---|---|---|---|
| Zacian (Metal) | 97% | 100% | 93% |
| Yveltal (Dark) | 87% | 83% | 97% |
| Latias (Psychic — our weakness) | 30% | 57% | **73%** |
| **Average** | 71% | 80% | **88%** |

## Layout

```
agents/          heuristic, search core/agent, opponent model, random baseline
decks/           deck lists (mega_lucario.py = main deck; gauntlet.py = eval opponents)
harness.py       seat-swapped match runner for two agents
run_gauntlet.py  3-way A/B: heuristic vs mirror-search vs modeled-search
make_submission.py  builds submission.tar.gz (main.py + deck.csv + modules at top level)
smoke_test.py    random-vs-random engine sanity check
submission/      main.py + deck.csv (Kaggle entrypoint)
```

## Setup

1. Download the simulator (SDK) and card data from the competition **Data** page.
2. Place the engine package at `cg/` in the repo root (so `from cg.api import ...` resolves).
3. `python smoke_test.py` to confirm the engine loads.
4. `python harness.py` or `python run_gauntlet.py` to evaluate.
5. `python make_submission.py` to build the upload bundle.

Requires Python 3.x. No third-party dependencies beyond the competition SDK.
