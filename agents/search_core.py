"""Determinized-rollout search core.

The competition hides the opponent's deck, hand, and prizes, plus our own deck/prize
ordering. `search_begin` needs concrete card-id lists for all of it, so we *determinize*:
sample one plausible world consistent with what we can see, run the real engine forward
under a rollout policy for a short horizon, and score the resulting board. Averaging the
score over several sampled worlds and candidate actions yields the move to play.

Isolation note: search uses `agent_ptr`; the live game uses `battle_ptr`. They don't
interfere, so this is safe to call from inside a competition agent.
"""
import random
from collections import Counter

from cg.api import (
    Observation, to_observation_class, all_card_data,
    AreaType, CardType, SelectType, SelectContext, OptionType,
    search_begin, search_step, search_end,
)

CARD_DB = {c.cardId: c for c in all_card_data()}
BASIC_ENERGY_FALLBACK = 6  # Basic Fighting Energy — pad id when a world runs short


def _known_ids(p, hide_hand=False):
    """Card ids at known locations for one player: hand (unless hidden), discard, in play."""
    ids = []
    if not hide_hand and p.hand:
        ids += [c.id for c in p.hand]
    ids += [c.id for c in p.discard]
    for area in (p.active, p.bench):
        for mon in area:
            if mon is None:
                continue
            ids.append(mon.id)
            ids += [e.id for e in mon.energyCards]
            ids += [t.id for t in mon.tools]
            ids += [pe.id for pe in mon.preEvolution]
    return ids


def determinize(obs: Observation, my_deck: list[int], opp_prior: list[int], rng: random.Random):
    """Sample one hidden-info world → args for search_begin.

    my_deck: our real 60-card decklist (we know it exactly).
    opp_prior: assumed opponent decklist (60 ids) used as a sampling prior.
    """
    st = obs.current
    me = st.yourIndex
    opp = 1 - me
    mp = st.players[me]
    op = st.players[opp]

    # our hidden cards (deck + prize) = decklist minus everything we can see
    rem = list((Counter(my_deck) - Counter(_known_ids(mp))).elements())
    rng.shuffle(rem)
    nd, npz = mp.deckCount, len(mp.prize)
    while len(rem) < nd + npz:
        rem.append(BASIC_ENERGY_FALLBACK)
    my_hidden_deck = rem[:nd]
    my_prize = rem[nd:nd + npz]

    # opponent hidden cards: sample from the prior minus what we've seen of theirs
    orem = list((Counter(opp_prior) - Counter(_known_ids(op, hide_hand=True))).elements())
    rng.shuffle(orem)
    need = op.deckCount + len(op.prize) + op.handCount
    while len(orem) < need:
        orem.append(BASIC_ENERGY_FALLBACK)
    orem = orem[:need]
    opp_deck = orem[:op.deckCount]
    opp_prize = orem[op.deckCount:op.deckCount + len(op.prize)]
    opp_hand = orem[op.deckCount + len(op.prize):]

    # opponent active id only needed while their active is face-down (setup)
    opp_active = []
    if op.active and op.active[0] is None:
        basic = next((cid for cid in opp_hand + opp_deck
                      if CARD_DB.get(cid) and CARD_DB[cid].cardType == CardType.POKEMON
                      and CARD_DB[cid].basic), None)
        opp_active = [basic] if basic else []

    return my_hidden_deck, my_prize, opp_deck, opp_prize, opp_hand, opp_active


# ---------------- value function ----------------

PRIZE_W = 120.0     # prizes remaining: dominant signal (0 remaining = win)
HP_W = 0.08
COUNT_W = 6.0
ENERGY_W = 1.5
HAND_W = 0.5
WIN_V = 100000.0


def value(obs: Observation, me: int) -> float:
    """Heuristic value of a rolled-out state, from seat `me`'s perspective."""
    st = obs.current
    if st is None:
        return 0.0
    if st.result != -1:
        if st.result == 2:
            return 0.0
        return WIN_V if st.result == me else -WIN_V

    mp = st.players[me]
    op = st.players[1 - me]

    v = 0.0
    # prizes remaining: fewer of mine is better
    v += PRIZE_W * (len(op.prize) - len(mp.prize))

    def board_hp(p):
        tot = 0
        for area in (p.active, p.bench):
            for mon in area:
                if mon:
                    tot += mon.hp
        return tot

    def in_play(p):
        return sum(1 for area in (p.active, p.bench) for mon in area if mon)

    def energy_on_board(p):
        tot = 0
        for area in (p.active, p.bench):
            for mon in area:
                if mon:
                    tot += len(mon.energies or [])
        return tot

    v += HP_W * (board_hp(mp) - board_hp(op))
    v += COUNT_W * (in_play(mp) - in_play(op))
    v += ENERGY_W * (energy_on_board(mp) - energy_on_board(op))
    v += HAND_W * (mp.handCount - op.handCount)
    return v
