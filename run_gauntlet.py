import sys, io, contextlib; sys.path.insert(0,'.')
from decks.mega_lucario import DECK as LUCARIO
from decks.gauntlet import GAUNTLET
from agents.search_agent import make_agent as search_agent
from agents.heuristic import make_agent as heur
from harness import run_match

def winrate(a0, a1, games):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        r=run_match(a0, a1, games, verbose=False)
    return r["wins"]

GAMES=30
print(f"Lucario agent vs gauntlet decks (heuristic-piloted opponents), {GAMES} games each")
print(f"{'opponent':<12} | {'heuristic':<12} | {'search-mirror':<14} | {'search-modeled':<14}")
print("-"*60)
for name, opp in GAUNTLET.items():
    h  = winrate(heur(LUCARIO), heur(opp), GAMES)
    sm = winrate(search_agent(LUCARIO, time_budget=0.15, horizon=3, model_opponent=False, seed=3), heur(opp), GAMES)
    so = winrate(search_agent(LUCARIO, time_budget=0.15, horizon=3, model_opponent=True,  seed=3), heur(opp), GAMES)
    def fmt(w): 
        d=w[0]+w[1]; return f"{w[0]}-{w[1]} ({w[0]/d*100 if d else 0:.0f}%)"
    print(f"{name:<12} | {fmt(h):<12} | {fmt(sm):<14} | {fmt(so):<14}", flush=True)
print("\nDONE")
