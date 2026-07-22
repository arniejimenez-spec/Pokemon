"""Gauntlet decks: distinct-type basic-ex attackers sharing one consistency shell.

Used only as *opponents* for local evaluation, so they favor simplicity/pilotability
by the heuristic over peak power. Each is a single big Basic ex attacker + a generic
big-basic search + draw shell + type energy.
"""

# shared trainer/support shell for a "one big basic attacker" deck (43 cards).
# Only cards confirmed legal in this competition format (Precious Trolley / Master Ball
# are out-of-format -> engine deck errorType=4).
SHELL = [
    (140, 2),   # Fezandipiti ex — colorless bench draw (Flip the Script)
    (1121, 4),  # Ultra Ball  (search any Pokemon)
    (1102, 4),  # Dusk Ball   (look bottom 7, reveal a Pokemon)
    (1231, 2),  # Dawn (search Basic + Stage1 + Stage2 — grabs our Basic attacker)
    (1097, 3),  # Night Stretcher (recover Pokemon/energy)
    (1123, 4),  # Switch
    (1174, 2),  # Air Balloon
    (1122, 3),  # Pokegear 3.0
    (1182, 4),  # Boss's Orders
    (1224, 4),  # Cheren (draw 3)
    (1208, 4),  # Iris's Fighting Spirit (discard 1, draw to 6)
    (1213, 2),  # Judge
    (1199, 1),  # Lacey
]  # = 43 cards


def build(attacker_id: int, energy_id: int, n_attacker: int = 4) -> list[int]:
    counts = list(SHELL) + [(attacker_id, n_attacker)]
    deck = [cid for cid, n in counts for _ in range(n)]
    deck += [energy_id] * (60 - len(deck))   # pad remainder with basic energy
    assert len(deck) == 60, f"{attacker_id}: {len(deck)} cards"
    return deck


# energy ids: G1 R2 W3 L4 P5 F6 D7 M8
# Chosen for type diversity + heuristic-pilotability (each crushes random ~100%).
ZACIAN = build(336, 8)          # Metal   (Slashing Strike 210)
YVELTAL = build(1062, 7)        # Dark    (Dark Strike 210)
LATIAS = build(184, 5)          # Psychic (Eon Blade 200) — tests Lucario's {P} weakness
TERAPAGOS = build(176, 4)       # Colorless/Lightning (Unified Beatdown, bench-scaling)

GAUNTLET = {
    "zacian": ZACIAN,
    "yveltal": YVELTAL,
    "latias": LATIAS,
    "terapagos": TERAPAGOS,
}
