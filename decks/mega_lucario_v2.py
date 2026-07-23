"""Deck v2 candidate: real-world competitive Mega Lucario (Lucario/Dudunsparce/Solrock).

Ported from the actual tournament-played "Mega Lucario" archetype on Limitless TCG
(https://limitlesstcg.com/decks/345, decklist https://limitlesstcg.com/decks/list/27961,
74th place NAIC 2026). Unlike the Dragapult/Dusknoir attempt, this deck's structure is
much closer to what our heuristic already handles well:

  - SAME primary attacker as our current deck: Riolu -> Mega Lucario ex (one Stage-1
    evolution line, proven to complete reliably ~14/15 games in earlier diagnostics).
  - A secondary evolution line (Dunsparce -> Dudunsparce/Dudunsparce ex) that is a
    SUPPORT piece (Run Away Draw: draw 3/turn), not a competing attacker -- a much
    lower-risk kind of "second line" than Dragapult's dual-attacker structure, where
    the heuristic had to choose which of two win conditions to commit to.
  - Solrock/Lunatone: a Basic utility pair, no evolution, minimal disruption timing.

Only one card from the real list is missing from our engine's card pool (Special Red
Card -- the same single gap found in the Dragapult and Ogerpon Box lists, suggesting
it's a very recent print not yet in the competition's card set); substituted with a
3rd Ultra Ball. Legacy Energy is this deck's ACE SPEC (replacing Prime Catcher).

Do not treat this as validated -- run the same heuristic-pilotability diagnostics used
on Dragapult (vs random, evolution-line completion rate, win-reason breakdown) before
investing in a full BC/RL retrain around it.
"""

DECK_LIST = [
    # Pokemon (17)
    (333, 4),   # Riolu
    (678, 3),   # Mega Lucario ex
    (65, 3),    # Dunsparce
    (66, 2),    # Dudunsparce (Run Away Draw: draw 3/turn)
    (306, 1),   # Dudunsparce ex
    (676, 2),   # Solrock
    (675, 2),   # Lunatone (Lunar Cycle synergises with Solrock)
    # Trainers (30)
    (1182, 4),  # Boss's Orders
    (1227, 4),  # Lillie's Determination
    (1225, 2),  # Hilda
    (1142, 4),  # Fighting Gong
    (1141, 4),  # Premium Power Pro
    (1152, 4),  # Poké Pad
    (1121, 3),  # Ultra Ball (bumped 2->3: substitute for missing Special Red Card)
    (1086, 2),  # Buddy-Buddy Poffin
    (1252, 2),  # Gravity Mountain
    (1246, 1),  # Jamming Tower
    # Energy (13)
    (6, 8),     # Basic Fighting Energy
    (20, 4),    # Rock Fighting Energy
    (12, 1),    # Legacy Energy (ACE SPEC)
]

DECK = [cid for cid, n in DECK_LIST for _ in range(n)]
assert len(DECK) == 60, len(DECK)
