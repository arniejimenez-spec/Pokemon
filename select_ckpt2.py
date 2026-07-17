import sys, io, contextlib; sys.path.insert(0,'.')
from decks.mega_lucario import DECK
from agents.policy_agent import make_agent as pol
from harness import run_match

def wr(a0,a1,n):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): r=run_match(a0,a1,n,verbose=False)
    w=r["wins"]; d=w[0]+w[1]
    return (w[0]/d if d else .5), f"{w[0]}-{w[1]}"

IT12="models/policy_rl_it12.npz"; N=300
print(f"Extended-RL selection: candidates vs frozen it12, {N} games each\n")
for k,p in [("it13","ckpt2/policy_it13.npz"),("it18","ckpt2/policy_it18.npz"),("it24","ckpt2/policy_it24.npz")]:
    r,s = wr(pol(DECK,model_path=p), pol(DECK,model_path=IT12), N)
    print(f"{k:5s} vs it12: {r:5.1%} ({s})", flush=True)
print("DONE")
