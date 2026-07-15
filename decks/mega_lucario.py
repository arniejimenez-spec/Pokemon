"""Deck v1: Mega Lucario ex.

Game plan: bench Riolu early (Poffin), evolve into Mega Lucario ex (Hyper Aroma is
ACE SPEC so we rely on Ultra/Dusk Ball + draw), alternate Mega Brave (270) between
two Lucarios or weave in Aura Jab (130 + reattach 3 F energy from discard).
"""

DECK_LIST = [
    # Pokemon (12)
    (333, 4),   # Riolu HP70 (Buddy-Buddy Poffin fetchable)
    (678, 4),   # Mega Lucario ex
    (140, 1),   # Fezandipiti ex (Flip the Script draw)
    (44, 2),    # Bloodmoon Ursaluna ex (late-game closer, cost shrinks)
    (979, 1),   # Koraidon ex (secondary F attacker, bench-safe Tera)
    # Items (19)
    (1086, 4),  # Buddy-Buddy Poffin
    (1121, 4),  # Ultra Ball
    (1102, 4),  # Dusk Ball
    (1097, 3),  # Night Stretcher
    (1088, 1),  # Prime Catcher (ACE SPEC)
    (1123, 3),  # Switch
    # Tools (2)
    (1174, 2),  # Air Balloon
    # Supporters (13)
    (1208, 4),  # Iris's Fighting Spirit (discard 1: draw to 6)
    (1224, 3),  # Cheren (draw 3)
    (1213, 1),  # Judge
    (1182, 3),  # Boss's Orders
    (1199, 1),  # Lacey
    (1211, 1),  # Black Belt's Training (+40 vs ex)
    # Energy (14)
    (6, 14),    # Basic Fighting Energy
]

DECK = [cid for cid, n in DECK_LIST for _ in range(n)]
assert len(DECK) == 60, len(DECK)
