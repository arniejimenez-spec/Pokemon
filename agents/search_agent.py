"""Determinized-rollout search agent.

Falls back to the heuristic for forced/trivial selections and uses budgeted,
anytime determinized rollouts at genuine decision points. The heuristic doubles
as the rollout policy for both players inside the search.
"""
import random
import time

from cg.api import (
    Observation, to_observation_class,
    SelectType, OptionType,
    search_begin, search_step, search_end,
)
try:  # local dev uses the agents package; the flat Kaggle bundle does not
    from agents.heuristic import HeuristicPolicy, ATTACK_DB
    from agents.search_core import determinize, value, CARD_DB
except ImportError:
    from heuristic import HeuristicPolicy, ATTACK_DB
    from search_core import determinize, value, CARD_DB


class SearchAgent:
    def __init__(self, deck: list[int], opp_prior: list[int] | None = None,
                 time_budget: float = 0.10, horizon: int = 3,
                 max_candidates: int = 4, min_rollouts_per_cand: int = 1,
                 model_opponent: bool = True, seed: int = 0):
        self.deck = deck
        self.static_opp_prior = opp_prior  # if set, overrides inference (mirror otherwise)
        self.opp_model = None
        if model_opponent and opp_prior is None:
            try:
                from agents.opponent_model import OpponentPrior
            except ImportError:
                from opponent_model import OpponentPrior
            self.opp_model = OpponentPrior(deck)
        self.policy = HeuristicPolicy(deck)
        self.time_budget = time_budget
        self.horizon = horizon
        self.max_candidates = max_candidates
        self.min_rollouts = min_rollouts_per_cand
        self.rng = random.Random(seed)

    # ---- entry point ----
    def act(self, obs: Observation) -> list[int]:
        sel = obs.select
        if sel is None:
            return self.deck
        opts = sel.option
        n = len(opts)
        lo, hi = sel.minCount, sel.maxCount

        # forced: single legal answer
        if n == 0:
            return []
        if lo == hi and hi == n:
            return list(range(n))
        if n == 1 and lo <= 1 <= hi:
            return [0]

        # Only invest search where a single index is chosen among several and the
        # decision is strategically meaningful (main line / attack / gust targets).
        if not self._is_searchable(obs):
            return self.policy.select(obs)

        return self._search(obs)

    def _is_searchable(self, obs: Observation) -> bool:
        sel = obs.select
        st = obs.current
        if st is None or st.result != -1:
            return False
        # setup / mulligan: determinization edge cases — let heuristic handle
        if st.turn <= 0:
            return False
        if sel.type not in (SelectType.MAIN, SelectType.ATTACK):
            return False
        # need a real branch and a single-pick decision
        if sel.maxCount != 1 or len(sel.option) < 2:
            return False
        return obs.search_begin_input is not None

    # ---- the search ----
    def _search(self, obs: Observation) -> list[int]:
        sel = obs.select
        me = obs.current.yourIndex

        # opponent prior for this decision: static override, inferred type, or mirror
        if self.static_opp_prior is not None:
            self._cur_prior = self.static_opp_prior
        elif self.opp_model is not None:
            self._cur_prior = self.opp_model.prior_for(obs, me)
        else:
            self._cur_prior = self.deck

        # candidate options: rank by the heuristic's own scoring, keep top-K + always
        # include END if present (so we can decide NOT to commit an attack)
        cand = self._candidates(obs)

        stats = {i: [0.0, 0] for i in cand}  # index -> [sum_value, n]
        start = time.perf_counter()
        deadline = start + self.time_budget
        # absolute cap: no new rollout starts after `deadline`, and rollouts
        # in flight abort at `hard_deadline`. Nothing overrides these — on slow
        # hardware we'd rather play the heuristic move than time out the episode.
        hard_deadline = start + self.time_budget * 2

        failures = 0
        done = False
        while not done:
            progressed = False
            for i in cand:  # cand is heuristic-ranked: best candidates sample first
                if time.perf_counter() >= deadline:
                    done = True
                    break
                v = self._rollout_option(obs, i, me, hard_deadline)
                if v is None:
                    failures += 1
                    # determinization/search keeps failing: bail to heuristic fast
                    if failures >= 2 * len(cand) and not any(c for _, c in stats.values()):
                        done = True
                        break
                else:
                    stats[i][0] += v
                    stats[i][1] += 1
                    progressed = True
            if not progressed and failures:
                break

        # self-tune to slow hardware: if this decision badly overran the budget,
        # shorten the horizon for the rest of the game
        elapsed = time.perf_counter() - start
        if elapsed > 3 * self.time_budget and self.horizon > 1:
            self.horizon -= 1

        # pick best mean value; fall back to heuristic's top choice on ties/empties
        best_i, best_v = None, -1e18
        for i in cand:
            s, c = stats[i]
            if c == 0:
                continue
            mean = s / c
            if mean > best_v:
                best_v, best_i = mean, i
        if best_i is None:
            return self.policy.select(obs)
        return [best_i]

    def _candidates(self, obs: Observation) -> list[int]:
        sel = obs.current
        opts = obs.select.option
        # score each option with the heuristic's per-option scorer where available
        me = obs.current.yourIndex
        my = obs.current.players[me]
        hand_ids = [c.id for c in (my.hand or [])]
        scored = []
        for i, o in enumerate(opts):
            if obs.select.type == SelectType.MAIN:
                s = self.policy._score_main_option(obs, i, o, hand_ids)
            else:  # ATTACK
                s = self.policy._effective_damage(obs, ATTACK_DB.get(o.attackId))
            scored.append((s, i))
        scored.sort(reverse=True)
        cand = [i for _, i in scored[:self.max_candidates]]
        # ensure END is considered so we can choose not to over-commit
        for i, o in enumerate(opts):
            if o.type == OptionType.END and i not in cand:
                cand.append(i)
        return cand

    def _rollout_option(self, obs: Observation, opt_index: int, me: int,
                        hard_deadline: float | None = None):
        """Apply `opt_index`, then roll out with the heuristic to the horizon; return value.

        Aborts (returning the partial-state value) once `hard_deadline` passes, so a
        single slow rollout can never blow the per-move time limit.
        """
        world = determinize(obs, self.deck, self._cur_prior, self.rng)
        try:
            root = search_begin(obs, *world)
        except (ValueError, RuntimeError):
            return None
        try:
            state = search_step(root.searchId, [opt_index])
            cur = state.observation
            start_turn = obs.current.turn
            steps = 0
            while cur.current is None or cur.current.result == -1:
                if cur.current and (cur.current.turn - start_turn) >= self.horizon:
                    break
                if hard_deadline is not None and time.perf_counter() >= hard_deadline:
                    break
                sel_list = self.policy.select(cur)
                state = search_step(state.searchId, sel_list)
                cur = state.observation
                steps += 1
                if steps > 400:
                    break
            return value(cur, me)
        except (ValueError, RuntimeError):
            return None
        finally:
            search_end()


def make_agent(deck: list[int], **kw):
    ag = SearchAgent(deck, **kw)

    def agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        try:
            return ag.act(obs)
        except Exception:
            # a crash forfeits the game on the ladder — always return a legal move
            if obs.select is None:
                return deck
            try:
                return ag.policy.select(obs)
            except Exception:
                sel = obs.select
                lo = sel.minCount
                return list(range(lo)) if lo > 0 else [0]

    return agent
