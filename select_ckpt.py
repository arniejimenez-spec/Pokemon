import sys, io, contextlib; sys.path.insert(0,'.')
from decks.mega_lucario import DECK
from agents.policy_agent import make_agent as pol
from agents.heuristic import make_agent as heur
from harness import run_match

def wr(a0,a1,n):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): r=run_match(a0,a1,n,verbose=False)
    w=r["wins"]; d=w[0]+w[1]
    return (w[0]/d if d else .5), f"{w[0]}-{w[1]}"

BC="agents/policy.npz"; N=200
cands={"it5":"ckpt/policy_it5.npz","it9":"ckpt/policy_it9.npz","it12":"ckpt/policy_it12.npz"}
print(f"Checkpoint selection, {N} games/pairing (breaks the winner's-curse tie)\n")
for k,p in cands.items():
    r1,s1 = wr(pol(DECK,model_path=p), pol(DECK,model_path=BC), N)
    r2,s2 = wr(pol(DECK,model_path=p), heur(DECK), N)
    print(f"{k:5s} vs BC: {r1:5.1%} ({s1:>7s})   vs heuristic: {r2:5.1%} ({s2:>7s})", flush=True)
print("\nhead-to-head:")
r,s = wr(pol(DECK,model_path=cands["it12"]), pol(DECK,model_path=cands["it5"]), N)
print(f"it12 vs it5: {r:.1%} ({s})", flush=True)
print("\nDONE")
