import sys, io, contextlib; sys.path.insert(0,'.')
from decks.mega_lucario import DECK
from agents.policy_agent import make_agent as pol
from harness import run_match

def wr(a0,a1,n):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): r=run_match(a0,a1,n,verbose=False)
    w=r["wins"]; d=w[0]+w[1]
    return (w[0]/d if d else .5), f"{w[0]}-{w[1]}"

IT18="models/policy_rl_it18.npz"; N=300
print(f"v2 lineage at EQUAL training (30 iters) vs champion it18, {N} games each\n")
for k,p in [("v2b-it5","ckpt_v2b/policy_it5.npz"),("v2b-it12","ckpt_v2b/policy_it12.npz"),("v2b-it18","ckpt_v2b/policy_it18.npz")]:
    r,s = wr(pol(DECK,model_path=p), pol(DECK,model_path=IT18), N)
    print(f"{k:9s} vs it18: {r:5.1%} ({s})", flush=True)
print("DONE")
