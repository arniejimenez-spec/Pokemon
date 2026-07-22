"""Diverse evaluation panel — the selection instrument that replaces mirror head-to-heads.

Why: a 57% Lucario-mirror edge shipped as a challenger and produced a 6.6-point
(= zero) ladder gap. Mirror selection optimizes something the ladder doesn't measure.
This panel scores a candidate against a VARIED set of opponents, including a deck
deliberately held out of RL training (Latias — also our weakness matchup), so panel
gains indicate generalization rather than self-familiarity.

Usage:
    python eval_panel.py models/policy_rl_v2_it30.npz ckpt_v3/policy_it8.npz --games 300
"""
import argparse
import io
import contextlib
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decks.mega_lucario import DECK as LUCARIO
from decks.gauntlet import GAUNTLET
from agents.policy_agent import make_agent as pol
from agents.heuristic import make_agent as heur
from harness import run_match

# name -> (make_opponent, note). Latias is HELD OUT of RL training on purpose.
def _panel(champion_path: str):
    champ_label = os.path.splitext(os.path.basename(champion_path))[0]
    return [
        ("heur-lucario", lambda: heur(LUCARIO), "mirror deck, scripted pilot"),
        ("zacian",       lambda: heur(GAUNTLET["zacian"]), "Metal, in training pool"),
        ("yveltal",      lambda: heur(GAUNTLET["yveltal"]), "Dark, in training pool"),
        ("latias",       lambda: heur(GAUNTLET["latias"]), "Psychic HOLDOUT + weakness"),
        (champ_label,    lambda: pol(LUCARIO, model_path=champion_path), "reigning champion (head-to-head)"),
    ]


def _wr(a0, a1, n):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = run_match(a0, a1, n, verbose=False)
    w = r["wins"]
    d = w[0] + w[1]
    return (w[0] / d if d else 0.5), w


def panel_score(model_path: str, games_per: int = 300,
                champion_path: str = "models/policy_rl_it18.npz") -> dict:
    rows = {}
    total_w = total_d = 0
    for name, make_opp, note in _panel(champion_path):
        r, w = _wr(pol(LUCARIO, model_path=model_path), make_opp(), games_per)
        rows[name] = (r, f"{w[0]}-{w[1]}", note)
        total_w += w[0]
        total_d += w[0] + w[1]
    rows["AGGREGATE"] = (total_w / total_d, f"{total_w}/{total_d}", "")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--games", type=int, default=300, help="games per panel opponent")
    ap.add_argument("--champion", default="models/policy_rl_it18.npz")
    args = ap.parse_args()

    for m in args.models:
        print(f"\n=== PANEL: {m} ({args.games} games/opponent) ===")
        rows = panel_score(m, args.games, args.champion)
        for name, (r, s, note) in rows.items():
            flag = "  <== HOLDOUT" if name == "latias" else ""
            print(f"  {name:20s} {r:6.1%}  ({s}) {note}{flag}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
