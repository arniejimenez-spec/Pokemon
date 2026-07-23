"""Feature encoding v3: v2 (option-identity embedding) + state-level id-bags.

Why v3 exists: v1/v2 encode board state via COUNTS only (hand size, discard size,
deck count) -- the network never sees WHAT is actually in hand, in the discard
piles, or likely still in the deck. The official competition sample notebook bags
exactly this kind of zone content (hand/decklist/discard) via a learned
EmbeddingBag. Two consecutive RL cycles on the v1/v2 architecture plateaued (see
EXPERIMENTS.md cycles 2-3: more iterations, then a wider opponent pool, both null)
-- this is the next genuinely different lever rather than more training on the
same representation.

v3 adds FOUR state-level "bags" (multisets of card ids, summed through a learned
embedding table, looked up in the model rather than here):
  - own hand
  - own deck-remaining: decklist MINUS every card whose location is currently
    known (hand + discard + in-play + attached) -- an EXACT residual, since we
    always know our own decklist. Prizes are not subtracted: they are also
    "unknown location", i.e. correctly lumped into this same bag.
  - own discard pile
  - opponent's discard pile
Both discard piles are directly visible in the Observation. We deliberately do
NOT attempt to model the opponent's hand or remaining deck -- that requires
guessing, and the search agent's opponent_model.py already showed that kind of
guesswork is not reliably net-positive. v3 only uses information that is exactly
known.

Bags are returned as raw card-id arrays here (not embedded) -- the embedding
table lives in the model (numpy for inference, torch for training) and is
looked up + summed there, exactly like features_v2's per-option identity.
"""
import numpy as np
from collections import Counter

from cg.api import Observation

try:
    from agents import features as _v1
    from agents.features_v2 import encode_option_ids, N_CARD_IDS
except ImportError:
    import features as _v1
    from features_v2 import encode_option_ids, N_CARD_IDS

FEATURE_VERSION = 3
STATE_EMB_DIM = 16   # dim of the shared state-bag embedding table

STATE_DIM = _v1.STATE_DIM
OPTION_DIM = _v1.OPTION_DIM
encode_state = _v1.encode_state
encode_options = _v1.encode_options


def _known_own_cards(state, player_index: int) -> list[int]:
    """Card ids whose location we know for certain: hand, discard, in-play
    (+ attached energy/tools/pre-evolution), our own Stadium if one is in play,
    and any card of ours currently in State.looking (e.g. mid-resolution of a
    'look at top N' effect). These are the only locations outside deck/prize a
    card can sit in, so missing any of them silently inflates deck-remaining.
    """
    p = state.players[player_index]
    ids = []
    if p.hand:
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
    for c in state.stadium:
        if c.playerIndex == player_index:
            ids.append(c.id)
    if state.looking:
        for c in state.looking:
            if c is not None and c.playerIndex == player_index:
                ids.append(c.id)
    return ids


def encode_state_bags(obs: Observation, my_deck: list[int]):
    """Return (hand_ids, deck_remaining_ids, own_discard_ids, opp_discard_ids),
    each a 1-D int32 array of raw card ids (possibly empty).

    Invariant: len(deck_remaining_ids) == mp.deckCount + len(mp.prize), true for
    ~96% of decisions (verified over 2192 decisions / 15 games). The residual is
    two known, small, self-correcting cases we deliberately don't chase further:
    (1) turn-0 setup, where your own Active pick can show as facedown (None) even
    to you until both players reveal simultaneously -- unrecoverable from a single
    stateless observation; (2) a one-decision transient during multi-step ATTACH
    resolutions, corrected on the very next decision.
    """
    st = obs.current
    me = st.yourIndex
    mp, op = st.players[me], st.players[1 - me]

    hand_ids = np.array([c.id for c in (mp.hand or [])], dtype=np.int32)

    known_ids = _known_own_cards(st, me)
    # A trainer card mid-resolution has already left hand but hasn't yet moved to
    # discard -- it sits in SelectData.effect / .contextCard for the duration of
    # its own effect (e.g. Ultra Ball, Switch). Missing this is the single most
    # common source of a stale deck-remaining count (~20% of decisions).
    sel = obs.select
    if sel is not None:
        for c in (sel.effect, sel.contextCard):
            if c is not None and c.playerIndex == me:
                known_ids.append(c.id)
    known = Counter(known_ids)
    remaining = list((Counter(my_deck) - known).elements())
    deck_remaining_ids = np.array(remaining, dtype=np.int32)

    own_discard_ids = np.array([c.id for c in mp.discard], dtype=np.int32)
    opp_discard_ids = np.array([c.id for c in op.discard], dtype=np.int32)

    return hand_ids, deck_remaining_ids, own_discard_ids, opp_discard_ids


def encode(obs: Observation, my_deck: list[int]):
    """Return (state, opts, option_ids, hand_ids, deck_remaining_ids,
    own_discard_ids, opp_discard_ids). `my_deck` is the 60-card decklist of
    whichever player is currently deciding (obs.current.yourIndex) -- required
    because deck-remaining can only be computed against a KNOWN decklist.
    """
    state, opts = encode_state(obs), encode_options(obs)
    option_ids = encode_option_ids(obs)
    bags = encode_state_bags(obs, my_deck)
    return (state, opts, option_ids) + bags
