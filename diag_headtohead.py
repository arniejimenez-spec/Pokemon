import sys, io, contextlib; sys.path.insert(0,'.')
from decks.mega_lucario import DECK as LUCARIO
from agents.search_agent import make_agent as sa
from harness import run_match

# Direct A vs B: same deck, same search, differ ONLY in opponent model.
# This is the strongest (most ladder-like) opponent I can produce locally.
def wr(a0, a1, n):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        r=run_match(a0, a1, n, verbose=False)
    return r["wins"]

N=40
print(f"Head-to-head, both Mega Lucario + horizon-3 search, {N} games seat-swapped")
modeled = sa(LUCARIO, time_budget=0.15, horizon=3, model_opponent=True, seed=1)
mirror  = sa(LUCARIO, time_budget=0.15, horizon=3, model_opponent=False, seed=2)
w = wr(modeled, mirror, N)
d = w[0]+w[1]
print(f"MODELED vs MIRROR: {w[0]}-{w[1]} draws {w[2]}  -> modeled win rate {w[0]/d*100 if d else 0:.0f}%")
print("(if mirror wins, opponent modeling is the regression)")
print("DONE")
