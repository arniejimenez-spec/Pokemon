"""Opponent modeling for determinization.

The mirror prior (assume the opponent runs our deck) is wrong against a diverse
ladder. Here we infer the opponent's dominant energy type from the Pokémon and
energy we can see, then hand back a plausible type-matched 60-card prior for the
hidden bulk (their deck/hand/prizes). Falls back to the mirror prior when we
haven't seen enough to guess.
"""
from collections import Counter

from cg.api import Observation, CardType, EnergyType

try:
    from agents.search_core import CARD_DB
except ImportError:
    from search_core import CARD_DB

# Self-contained per-type prior template (no dependency on the local decks/ package,
# so this ships in the flat Kaggle bundle). Same "one big Basic attacker + generic
# consistency shell + type energy" pattern used by the eval gauntlet.
_SHELL = [
    (140, 2), (1121, 4), (1102, 4), (1231, 2), (1097, 3), (1123, 4),
    (1174, 2), (1122, 3), (1182, 4), (1224, 4), (1208, 4), (1213, 2), (1199, 1),
]


def _template(attacker_id: int, energy_id: int) -> list[int]:
    deck = [cid for cid, n in _SHELL for _ in range(n)] + [attacker_id] * 4
    return deck + [energy_id] * (60 - len(deck))


# Representative mono-energy Basic-ex attacker per type. Priors only, not pilotable
# decks. Water/Lightning omitted (no clean mono-energy Basic-ex in pool) -> those
# opponents fall back to the mirror prior.
_TYPE_TEMPLATE = {
    EnergyType.FIRE:     _template(46, 2),    # Gouging Fire ex
    EnergyType.PSYCHIC:  _template(184, 5),   # Latias ex
    EnergyType.FIGHTING: _template(979, 6),   # Koraidon ex
    EnergyType.DARKNESS: _template(1062, 7),  # Yveltal ex
    EnergyType.METAL:    _template(336, 8),   # Zacian ex
    EnergyType.GRASS:    _template(75, 1),    # Iron Leaves ex
}


def infer_opponent_type(obs: Observation, me: int) -> EnergyType | None:
    """Dominant energy type of the opponent from visible Pokémon + attached energy."""
    st = obs.current
    if st is None:
        return None
    op = st.players[1 - me]
    votes: Counter = Counter()

    def add_mon(mon, weight):
        if mon is None:
            return
        c = CARD_DB.get(mon.id)
        if c and c.cardType == CardType.POKEMON:
            et = c.energyType
            if et not in (EnergyType.COLORLESS, EnergyType.RAINBOW):
                votes[et] += weight
        for e in (mon.energies or []):
            if e not in (EnergyType.COLORLESS, EnergyType.RAINBOW):
                votes[e] += weight  # attached energy is a strong signal

    for mon in op.active:
        add_mon(mon, 3)          # active attacker is the best signal
    for mon in op.bench:
        add_mon(mon, 2)
    for c in op.discard:         # discarded Pokémon/energy also hint at type
        cd = CARD_DB.get(c.id)
        if cd and cd.cardType == CardType.POKEMON and cd.energyType not in (
                EnergyType.COLORLESS, EnergyType.RAINBOW):
            votes[cd.energyType] += 1
        elif cd and cd.cardType == CardType.BASIC_ENERGY and cd.energyType not in (
                EnergyType.COLORLESS, EnergyType.RAINBOW):
            votes[cd.energyType] += 1

    if not votes:
        return None
    return votes.most_common(1)[0][0]


class OpponentPrior:
    """Chooses a determinization prior deck for the opponent, cached per type guess."""

    def __init__(self, my_deck: list[int]):
        self.mirror = list(my_deck)

    def prior_for(self, obs: Observation, me: int) -> list[int]:
        et = infer_opponent_type(obs, me)
        if et is not None and et in _TYPE_TEMPLATE:
            return _TYPE_TEMPLATE[et]
        return self.mirror
