import sys, io, contextlib; sys.path.insert(0,'.')
from decks.mega_lucario import DECK as LUCARIO
from decks.gauntlet import GAUNTLET
from agents.search_agent import make_agent as sa
from agents.heuristic import make_agent as heur
from harness import run_match

def wr(a0,a1,n):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): r=run_match(a0,a1,n,verbose=False)
    return r["wins"]

w = wr(sa(LUCARIO, time_budget=0.10, seed=9), heur(LUCARIO), 30)
print(f"hardened-search vs heuristic (mirror decks): {w[0]}-{w[1]} ({w[0]/(w[0]+w[1])*100:.0f}%)", flush=True)
w = wr(sa(LUCARIO, time_budget=0.10, seed=9), heur(GAUNTLET['latias']), 30)
print(f"hardened-search vs latias: {w[0]}-{w[1]} ({w[0]/(w[0]+w[1])*100:.0f}%)", flush=True)
print("DONE")
