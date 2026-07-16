"""Opponent modeling for determinization.

v1 used weak per-type template decks as priors — that backfired on the ladder:
modeling the opponent as a vanilla pile makes rollouts underestimate real threats,
and observed opponent cards that aren't in the template create inconsistent worlds
(failed determinizations degrade the agent to heuristic-quality moves).

v2 keeps the strength of the mirror prior but adapts it with evidence:

  prior = every opponent card we've actually observed
        + copies of their observed Pokémon lines (assume they run 3 of each)
        + our own deck's trainer skeleton as generic strong filler
        + basic energy matching their observed dominant type

Observed cards are in the prior by construction, so the determinizer's
subtraction (prior − observed) is always consistent.
"""
from collections import Counter

from cg.api import Observation, CardType, EnergyType

try:
    from agents.search_core import CARD_DB
except ImportError:
    from search_core import CARD_DB

# Basic energy card id per EnergyType (G=1 R=2 W=3 L=4 P=5 F=6 D=7 M=8)
_ENERGY_CARD = {
    EnergyType.GRASS: 1, EnergyType.FIRE: 2, EnergyType.WATER: 3,
    EnergyType.LIGHTNING: 4, EnergyType.PSYCHIC: 5, EnergyType.FIGHTING: 6,
    EnergyType.DARKNESS: 7, EnergyType.METAL: 8,
}


def observed_opponent_cards(obs: Observation, me: int) -> Counter:
    """Every opponent card id we can currently see (board, attachments, discard)."""
    op = obs.current.players[1 - me]
    seen: Counter = Counter()
    for c in op.discard:
        seen[c.id] += 1
    for area in (op.active, op.bench):
        for mon in area:
            if mon is None:
                continue
            seen[mon.id] += 1
            for e in mon.energyCards:
                seen[e.id] += 1
            for t in mon.tools:
                seen[t.id] += 1
            for pe in mon.preEvolution:
                seen[pe.id] += 1
    return seen


def infer_opponent_type(obs: Observation, me: int) -> EnergyType | None:
    """Dominant energy type of the opponent from visible Pokémon + attached energy."""
    st = obs.current
    if st is None:
        return None
    op = st.players[1 - me]
    votes: Counter = Counter()

    def vote(et, w):
        if et is not None and et not in (EnergyType.COLORLESS, EnergyType.RAINBOW):
            votes[et] += w

    for weight, area in ((3, op.active), (2, op.bench)):
        for mon in area:
            if mon is None:
                continue
            c = CARD_DB.get(mon.id)
            if c and c.cardType == CardType.POKEMON:
                vote(c.energyType, weight)
            for e in (mon.energies or []):
                vote(e, weight)
    for c in op.discard:
        cd = CARD_DB.get(c.id)
        if cd and cd.cardType in (CardType.POKEMON, CardType.BASIC_ENERGY):
            vote(cd.energyType, 1)

    if not votes:
        return None
    return votes.most_common(1)[0][0]


class OpponentPrior:
    """Evidence-adapted strong prior for the opponent's 60 cards."""

    def __init__(self, my_deck: list[int]):
        self.mirror = list(my_deck)
        # our trainer skeleton (non-Pokemon, non-energy) = generic strong filler
        self.trainer_skeleton = [
            cid for cid in my_deck
            if (c := CARD_DB.get(cid)) and c.cardType in (
                CardType.ITEM, CardType.SUPPORTER, CardType.TOOL, CardType.STADIUM)
        ]

    def prior_for(self, obs: Observation, me: int) -> list[int]:
        if obs.current is None:
            return self.mirror

        seen = observed_opponent_cards(obs, me)
        if not seen:
            return self.mirror

        prior: list[int] = list(seen.elements())

        # assume ~3 copies of each observed opponent Pokemon line (cap at 4 total)
        for cid, n in seen.items():
            c = CARD_DB.get(cid)
            if c and c.cardType == CardType.POKEMON:
                prior += [cid] * max(0, min(4, n + 2) - n)

        # energy filler in their dominant type
        et = infer_opponent_type(obs, me)
        energy_id = _ENERGY_CARD.get(et, 6)
        n_energy_seen = sum(n for cid, n in seen.items()
                            if (c := CARD_DB.get(cid)) and c.cardType == CardType.BASIC_ENERGY)
        prior += [energy_id] * max(0, 12 - n_energy_seen)

        # top up to 60 with our trainer skeleton (generic competitive filler)
        for cid in self.trainer_skeleton:
            if len(prior) >= 60:
                break
            prior.append(cid)
        # still short (tiny skeleton?) -> pad energy
        prior += [energy_id] * (60 - len(prior))
        return prior[:60]
