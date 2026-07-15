"""Heuristic baseline agent.

Plays a sensible priority-driven game: bench basics, attach energy to attackers,
evolve, use draw, attack for max damage (preferring KOs). Deck-agnostic core with
per-deck priority hints.
"""
from cg.api import (
    Observation, to_observation_class, all_card_data, all_attack,
    AreaType, CardType, SelectType, SelectContext, OptionType, EnergyType,
)

CARD_DB = {c.cardId: c for c in all_card_data()}
ATTACK_DB = {a.attackId: a for a in all_attack()}


# ---------- helpers ----------

def get_pokemon(state, player_index, area, index):
    p = state.players[player_index]
    if area == AreaType.ACTIVE:
        return p.active[index] if index < len(p.active) else None
    if area == AreaType.BENCH:
        return p.bench[index] if index < len(p.bench) else None
    return None


def resolve_card_id(obs, opt):
    """Best-effort card id for a CARD-type option. None if facedown/unknown."""
    state = obs.current
    sel = obs.select
    area = opt.area
    idx = opt.index
    if area is None or idx is None:
        return None
    if area == AreaType.DECK:
        if sel.deck is not None and idx < len(sel.deck):
            return sel.deck[idx].id
        return None
    pi = opt.playerIndex if opt.playerIndex is not None else state.yourIndex
    p = state.players[pi]
    try:
        if area == AreaType.HAND:
            return p.hand[idx].id if p.hand else None
        if area == AreaType.ACTIVE:
            mon = p.active[idx]
            return mon.id if mon else None
        if area == AreaType.BENCH:
            return p.bench[idx].id
        if area == AreaType.DISCARD:
            return p.discard[idx].id
        if area == AreaType.PRIZE:
            c = p.prize[idx]
            return c.id if c else None
        if area == AreaType.STADIUM:
            return state.stadium[idx].id
        if area == AreaType.LOOKING:
            c = state.looking[idx]
            return c.id if c else None
    except (IndexError, TypeError):
        return None
    return None


def best_attack_damage(card_id, energies_attached=None):
    """Max printed damage among a card's attacks (usable ignore-cost upper bound)."""
    card = CARD_DB.get(card_id)
    if not card or not card.attacks:
        return 0
    return max(ATTACK_DB[a].damage for a in card.attacks if a in ATTACK_DB)


def energy_count(mon):
    return len(mon.energies) if mon and mon.energies else 0


def cheapest_attack_cost(card_id):
    card = CARD_DB.get(card_id)
    if not card or not card.attacks:
        return 99
    return min(len(ATTACK_DB[a].energies) for a in card.attacks if a in ATTACK_DB)


class HeuristicPolicy:
    """Encapsulates choice logic; subclass or pass hints to specialize per deck."""

    def __init__(self, deck: list[int], acquire_priority: dict[int, float] | None = None):
        self.deck = deck
        self.deck_set = set(deck)
        # score for acquiring a card into hand/bench via searches
        self.acquire = acquire_priority or {}
        # derive defaults: evolutions of our deck's basics score high
        for cid in self.deck_set:
            c = CARD_DB.get(cid)
            if not c:
                continue
            if cid not in self.acquire:
                if c.cardType == CardType.POKEMON:
                    self.acquire[cid] = 50 + best_attack_damage(cid) / 10 + (20 if not c.basic else 0)
                elif c.cardType in (CardType.SUPPORTER, CardType.ITEM, CardType.TOOL):
                    self.acquire[cid] = 30
                else:
                    self.acquire[cid] = 20  # energy

    # ---- generic keep-value of a card in hand (higher = keep) ----
    def keep_value(self, card_id, hand_ids=None):
        c = CARD_DB.get(card_id)
        if c is None:
            return 0
        base = self.acquire.get(card_id, 25)
        if c.cardType == CardType.BASIC_ENERGY and hand_ids is not None:
            n_energy = sum(1 for h in hand_ids if CARD_DB.get(h) and CARD_DB[h].cardType == CardType.BASIC_ENERGY)
            if n_energy > 2:
                base = 5  # spare energy is the cheapest discard
        return base

    def select(self, obs: Observation) -> list[int]:
        sel = obs.select
        state = obs.current
        me = state.yourIndex if state else 0
        opts = sel.option
        n = len(opts)
        lo, hi = sel.minCount, sel.maxCount
        ctx = sel.context
        stype = sel.type

        # ----- YES/NO decisions -----
        if stype == SelectType.YES_NO:
            yes = next((i for i, o in enumerate(opts) if o.type == OptionType.YES), 0)
            no = next((i for i, o in enumerate(opts) if o.type == OptionType.NO), 0)
            if ctx == SelectContext.MULLIGAN:
                return [self._mulligan_choice(obs, yes, no)]
            if ctx == SelectContext.IS_FIRST:
                return [yes]
            # default: activate beneficial effects
            return [yes]

        # ----- COUNT: take max (draw etc.) -----
        if stype == SelectType.COUNT:
            best = max(range(n), key=lambda i: opts[i].number or 0)
            return [best]

        # ----- MAIN action -----
        if stype == SelectType.MAIN:
            return [self._main_action(obs)]

        # ----- ATTACK choice -----
        if stype == SelectType.ATTACK:
            return [self._pick_attack(obs)]

        # ----- CARD selections -----
        if stype in (SelectType.CARD, SelectType.CARD_OR_ATTACHED_CARD, SelectType.ATTACHED_CARD):
            return self._pick_cards(obs)

        if stype == SelectType.ENERGY:
            # discarding / moving energy: just take the first valid combination
            count = max(lo, min(hi, lo if lo > 0 else hi))
            return list(range(count))

        if stype == SelectType.EVOLVE:
            return list(range(max(lo, 1)))[:max(lo, 1)] if n else []

        # skill order, special condition, fallback: first legal
        count = lo if lo > 0 else min(1, hi)
        return list(range(count))

    # ---- specific deciders ----

    def _mulligan_choice(self, obs, yes, no):
        # keep if we have a basic we actually want; the engine only offers mulligan
        # when allowed. Redraw if hand has no basic Pokemon of our main line.
        state = obs.current
        me = state.yourIndex
        hand = state.players[me].hand or []
        basics = [c for c in hand if CARD_DB.get(c.id) and CARD_DB[c.id].cardType == CardType.POKEMON and CARD_DB[c.id].basic]
        return no if basics else yes

    def _main_action(self, obs) -> int:
        sel = obs.select
        state = obs.current
        me = state.yourIndex
        opp = 1 - me
        my = state.players[me]
        hand_ids = [c.id for c in (my.hand or [])]

        scored = []
        for i, o in enumerate(sel.option):
            s = self._score_main_option(obs, i, o, hand_ids)
            scored.append((s, i))
        scored.sort(reverse=True)
        return scored[0][1]

    def _score_main_option(self, obs, i, o, hand_ids) -> float:
        state = obs.current
        me = state.yourIndex
        opp = 1 - me
        my = state.players[me]
        t = o.type

        if t == OptionType.ATTACH and not state.energyAttached:
            # manual energy attach: prefer target that still needs energy for an attack
            target = get_pokemon(state, me, o.inPlayArea, o.inPlayIndex)
            if target:
                need = cheapest_attack_cost(target.id) - energy_count(target)
                dmg = best_attack_damage(target.id)
                return 90 + dmg / 10 + (15 if need > 0 else -10)
            return 80

        if t == OptionType.EVOLVE:
            card = CARD_DB.get(resolve_card_id(obs, o) or 0)
            return 85

        if t == OptionType.ABILITY:
            return 70  # abilities in our decks are draw/search: use them

        if t == OptionType.PLAY:
            cid = hand_ids[o.index] if o.index is not None and o.index < len(hand_ids) else None
            return self._score_play(obs, cid)

        if t == OptionType.ATTACK:
            atk = ATTACK_DB.get(o.attackId)
            dmg = self._effective_damage(obs, atk)
            opp_active = state.players[opp].active
            opp_hp = opp_active[0].hp if opp_active and opp_active[0] else 999
            ko = dmg >= opp_hp
            # attacking generally ends the turn: do it only after other actions,
            # but MAIN reappears after each action, so give attack modest base
            return 40 + dmg / 10 + (25 if ko else 0)

        if t == OptionType.RETREAT:
            return 5  # rarely: only if active is bad — keep low for baseline

        if t == OptionType.DISCARD:
            return 4

        if t == OptionType.END:
            return 0
        return 1

    def _score_play(self, obs, cid) -> float:
        if cid is None:
            return 20
        c = CARD_DB.get(cid)
        if c is None:
            return 20
        state = obs.current
        me = state.yourIndex
        my = state.players[me]
        hand_n = len(my.hand or [])
        if c.cardType == CardType.POKEMON and c.basic:
            bench_room = my.benchMax - len(my.bench)
            if bench_room <= 0:
                return 0
            in_play = len(my.bench) + len([a for a in my.active if a])
            return 110 if in_play <= 2 else 75
        if c.cardType == CardType.ITEM:
            return 65
        if c.cardType == CardType.TOOL:
            return 55
        if c.cardType == CardType.SUPPORTER:
            if state.supporterPlayed:
                return 0
            # draw supporters better with small hand; Boss's Orders when opponent
            # has a juicy bench target we might KO
            return 60 if hand_n <= 5 else 45
        if c.cardType == CardType.STADIUM:
            return 30 if not state.stadiumPlayed else 0
        return 20

    def _effective_damage(self, obs, atk) -> float:
        if atk is None:
            return 0
        state = obs.current
        me = state.yourIndex
        opp = 1 - me
        dmg = atk.damage
        my_active = state.players[me].active
        opp_active = state.players[opp].active
        if my_active and my_active[0] and opp_active and opp_active[0]:
            mc = CARD_DB.get(my_active[0].id)
            oc = CARD_DB.get(opp_active[0].id)
            if mc and oc and oc.weakness is not None and oc.weakness == mc.energyType:
                dmg *= 2
        return dmg

    def _pick_attack(self, obs) -> int:
        sel = obs.select
        best, best_score = 0, -1
        state = obs.current
        opp = 1 - state.yourIndex
        opp_active = state.players[opp].active
        opp_hp = opp_active[0].hp if opp_active and opp_active[0] else 999
        for i, o in enumerate(sel.option):
            atk = ATTACK_DB.get(o.attackId) if o.attackId is not None else None
            dmg = self._effective_damage(obs, atk)
            score = dmg + (500 if dmg >= opp_hp else 0)
            if score > best_score:
                best, best_score = i, score
        return best

    def _pick_cards(self, obs) -> list[int]:
        sel = obs.select
        state = obs.current
        me = state.yourIndex
        ctx = sel.context
        opts = sel.option
        lo, hi = sel.minCount, sel.maxCount

        beneficial = ctx in (
            SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON,
            SelectContext.TO_ACTIVE, SelectContext.TO_BENCH, SelectContext.TO_FIELD,
            SelectContext.TO_HAND, SelectContext.SWITCH, SelectContext.ATTACH_TO,
            SelectContext.ATTACH_FROM, SelectContext.HEAL,
            SelectContext.REMOVE_DAMAGE_COUNTER, SelectContext.EVOLVES_FROM,
            SelectContext.EVOLVES_TO, SelectContext.LOOK,
        )
        harmful_own = ctx in (
            SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM,
            SelectContext.TO_PRIZE, SelectContext.DISCARD_ENERGY_CARD,
            SelectContext.DISCARD_TOOL_CARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
        )
        offensive = ctx in (
            SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY,
            SelectContext.DAMAGE, SelectContext.EFFECT_TARGET, SelectContext.DEVOLVE,
        )

        def opt_score(o):
            cid = resolve_card_id(obs, o)
            pi = o.playerIndex
            mine = (pi is None or pi == me)
            if offensive and not mine:
                # target opponent: prefer active/low HP (finish KOs)
                mon = get_pokemon(state, pi, o.area, o.index) if o.area in (AreaType.ACTIVE, AreaType.BENCH) else None
                hp = mon.hp if mon else 100
                return 200 - hp
            if ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON,
                       SelectContext.TO_BENCH, SelectContext.TO_FIELD):
                return self.acquire.get(cid, 10) if cid else 10
            if ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                mon = get_pokemon(state, me, o.area, o.index)
                if mon:
                    need = cheapest_attack_cost(mon.id) - energy_count(mon)
                    return best_attack_damage(mon.id) + (100 if need <= 0 else 0) + mon.hp / 10
                return 10
            if ctx == SelectContext.TO_HAND or beneficial:
                score = self.acquire.get(cid, 10) if cid else 10
                # board thin: a basic in hand beats an evolution we can't play
                c = CARD_DB.get(cid) if cid else None
                if c and c.cardType == CardType.POKEMON and c.basic:
                    my = state.players[me]
                    in_play = len(my.bench) + len([a for a in my.active if a])
                    if in_play <= 2:
                        score += 80
                return score
            if harmful_own and mine:
                hand_ids = [c.id for c in (state.players[me].hand or [])]
                return -self.keep_value(cid, hand_ids) if cid else -10
            return 0

        ranked = sorted(range(len(opts)), key=lambda i: opt_score(opts[i]), reverse=True)
        # how many to take: max for beneficial/offensive, min for harmful
        if harmful_own:
            count = lo
        else:
            count = hi
        count = max(lo, min(count, hi))
        return ranked[:count] if count > 0 else []


def make_agent(deck: list[int]):
    policy = HeuristicPolicy(deck)

    def agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        return policy.select(obs)

    return agent
