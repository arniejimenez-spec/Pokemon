"""Deck v2 candidate: Dragapult ex / Dusknoir.

Ported from a real-world tournament decklist (2nd place, NAIC 2026 New Orleans,
piloted by Neddy Kosek; https://limitlesstcg.com/decks/list/28236). Nearly the
entire list maps directly onto our card pool -- only one 1-of tech
("Special Red Card") isn't in our engine's card set; substituted with a 3rd
Boss's Orders to keep the trainer count at 33.

Game plan: Dreepy -> Drakloak -> Dragapult ex (Stage 2, 320 HP, Phantom Dive 200)
as the primary attacker, with a second Stage 2 line (Duskull -> Dusclops ->
Dusknoir, Shadow Bind 150 flat) as a secondary/disruption threat. Two evolution
lines + heavy item disruption (Crushing Hammer, Unfair Stamp) makes this
mechanically more complex than Mega Lucario -- more sequencing decisions per
game, which may matter for how well our heuristic/BC/RL stack can pilot it.
"""

DECK_LIST = [
    # Pokemon (19)
    (119, 4),    # Dreepy
    (120, 4),    # Drakloak
    (121, 2),    # Dragapult ex
    (131, 2),    # Duskull
    (132, 2),    # Dusclops
    (133, 1),    # Dusknoir
    (235, 1),    # Budew
    (140, 1),    # Fezandipiti ex
    (1071, 1),   # Meowth ex
    (112, 1),    # Munkidori
    # Trainers (33)
    (1227, 4),   # Lillie's Determination
    (1198, 3),   # Crispin
    (1182, 3),   # Boss's Orders (bumped 2->3: substitute for missing Special Red Card)
    (1231, 1),   # Dawn
    (1121, 4),   # Ultra Ball
    (1152, 4),   # Poké Pad
    (1086, 4),   # Buddy-Buddy Poffin
    (1120, 4),   # Crushing Hammer
    (1097, 2),   # Night Stretcher
    (1080, 1),   # Unfair Stamp (ACE SPEC)
    (1161, 1),   # Handheld Fan
    (1256, 1),   # Team Rocket's Watchtower
    (1246, 1),   # Jamming Tower
    # Energy (8)
    (5, 3),      # Basic Psychic Energy
    (2, 3),      # Basic Fire Energy
    (7, 2),      # Basic Darkness Energy
]

DECK = [cid for cid, n in DECK_LIST for _ in range(n)]
assert len(DECK) == 60, len(DECK)
