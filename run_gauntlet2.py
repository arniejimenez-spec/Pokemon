import sys, io, contextlib; sys.path.insert(0,'.')
from decks.mega_lucario import DECK as LUCARIO
from decks.gauntlet import GAUNTLET
from agents.search_agent import make_agent as sa
from agents.heuristic import make_agent as heur
from harness import run_match

def wr(a0,a1,n):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): r=run_match(a0,a1,n,verbose=False)
    w=r["wins"]; d=w[0]+w[1]
    return f"{w[0]}-{w[1]} ({w[0]/d*100 if d else 0:.0f}%)"

N=30
print(f"v2 evidence-adapted prior vs mirror prior, {N} games/cell (hardened timing, budget 100ms)")
for name, opp in GAUNTLET.items():
    m2 = wr(sa(LUCARIO, time_budget=0.10, model_opponent=True,  seed=4), heur(opp), N)
    mi = wr(sa(LUCARIO, time_budget=0.10, model_opponent=False, seed=4), heur(opp), N)
    print(f"{name:<10} | modeled-v2 {m2:<14} | mirror {mi}", flush=True)
hh = wr(sa(LUCARIO, time_budget=0.10, model_opponent=True, seed=5),
        sa(LUCARIO, time_budget=0.10, model_opponent=False, seed=6), N)
print(f"head-to-head modeled-v2 vs mirror (same deck): {hh}", flush=True)
print("DONE")
