"""Self-play RL fine-tuning of the option-scoring policy (REINFORCE + value baseline).

Why this exists
---------------
Behavioural cloning caps us at "imitates a 601 heuristic". To beat it the policy has
to be pushed toward actions that *win*, not actions the heuristic would have taken.

Method
------
Start from the BC weights (a random policy would flail forever in a game this long),
then iterate:

  1. self-play games with the CURRENT policy, sampling actions at temperature>0
  2. reward = +1 win / -1 loss / 0 draw, credited to every decision that player made
  3. advantage = reward - V(state);  loss = -logpi(a|s)*advantage + value_loss
  4. export new weights, evaluate, keep the best checkpoint

Opponents are drawn from a POOL (current policy, past checkpoints, the heuristic)
rather than only self, so we don't overfit to our own quirks and forget how to beat
anything else.

Everything runs locally: 130k params trains in ~a minute on CPU, and self-play is
CPU-bound anyway, so a GPU round-trip would cost more than it saves.

Usage:
    python train/rl_loop.py --iters 10 --games 1200 --workers 7
"""
import argparse
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ───────────────────────── self-play data generation ─────────────────────────

def _selfplay(args):
    """Worker: play games with the current policy, record every decision + outcome."""
    n_games, seed, model_path, temperature, opp_kind = args

    from decks.mega_lucario import DECK as LUCARIO
    from decks.gauntlet import GAUNTLET
    from agents.policy_agent import PolicyAgent
    from agents.heuristic import HeuristicPolicy
    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import to_observation_class

    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)

    learner = PolicyAgent(LUCARIO, model_path=model_path)
    if learner.net is None:
        return None
    fv = learner.net.fv   # encode/logits dispatch on the model's feature version

    # opponent pool: our own policy (self-play), the heuristic, or a gauntlet deck
    if opp_kind == "self":
        opp_deck = LUCARIO
        opp = PolicyAgent(LUCARIO, model_path=model_path)
    elif opp_kind == "heuristic":
        opp_deck = LUCARIO
        opp = HeuristicPolicy(LUCARIO)
    else:
        opp_deck = GAUNTLET[opp_kind]
        opp = HeuristicPolicy(opp_deck)

    states, opts, ptr, ys = [], [], [0], []
    opt_ids = []
    rewards = []

    for _ in range(n_games):
        obs, sd = battle_start(LUCARIO, opp_deck)
        if obs is None:
            continue
        g0 = len(ys)
        steps = 0
        while obs["current"]["result"] == -1:
            o = to_observation_class(obs)
            sel = o.select
            seat = o.current.yourIndex
            n_opt = len(sel.option)

            if seat == 0:
                learnable = (n_opt > 1 and sel.minCount == 1 and sel.maxCount == 1)
                if learnable:
                    s, om, ids = learner.net.encode(o)
                    z = learner.net.logits(s, om, ids)
                    if not np.all(np.isfinite(z)):
                        play = learner.fallback.select(o)
                    else:
                        # sample for exploration; record what we actually played
                        p = np.exp((z - z.max()) / max(temperature, 1e-3))
                        p /= p.sum()
                        a = int(rng.choice(n_opt, p=p))
                        states.append(s); opts.append(om)
                        if fv == 2:
                            opt_ids.append(ids)
                        ptr.append(ptr[-1] + n_opt); ys.append(a)
                        play = [a]
                else:
                    play = learner.select(o)
            else:
                play = opp.select(o)

            obs = battle_select(play)
            steps += 1
            if steps > 3000:
                break

        result = obs["current"]["result"]
        battle_finish()
        r = 0.0 if result == 2 else (1.0 if result == 0 else -1.0)  # seat 0 = learner
        rewards.extend([r] * (len(ys) - g0))

    if not ys:
        return None
    return (np.stack(states).astype(np.float32),
            np.concatenate(opts).astype(np.float32),
            np.array(ptr, dtype=np.int64),
            np.array(ys, dtype=np.int16),
            np.array(rewards, dtype=np.float32),
            (np.concatenate(opt_ids).astype(np.int64) if fv == 2 else None))


def generate(model_path, n_games, workers, temperature, seed):
    from multiprocessing import Pool
    # mix opponents so we don't overfit to self-play quirks
    kinds = ["self", "self", "heuristic", "zacian", "yveltal", "latias", "self"]
    per = max(1, n_games // workers)
    jobs = [(per, seed + w * 7919, model_path, temperature, kinds[w % len(kinds)])
            for w in range(workers)]
    with Pool(workers) as pool:
        res = [r for r in pool.map(_selfplay, jobs) if r is not None]
    if not res:
        raise SystemExit("no self-play data")
    states = np.concatenate([r[0] for r in res])
    ys = np.concatenate([r[3] for r in res])
    rew = np.concatenate([r[4] for r in res])
    opt_list, ptr_list, off = [], [0], 0
    for r in res:
        opt_list.append(r[1])
        ptr_list.extend((r[2][1:] + off).tolist())
        off += r[1].shape[0]
    ids = (np.concatenate([r[5] for r in res]) if res[0][5] is not None else None)
    return states, np.concatenate(opt_list), np.array(ptr_list, dtype=np.int64), ys, rew, ids


# ───────────────────────────── torch model I/O ───────────────────────────────

def load_torch(model_path):
    import torch.nn as nn
    import torch
    z = np.load(model_path)
    n = int(z["n_layers"][0])
    mods, dims = [], []
    for i in range(n):
        W, b = z[f"W{i}"], z[f"b{i}"]
        lin = nn.Linear(W.shape[0], W.shape[1])
        lin.weight.data = torch.tensor(W.T.copy())
        lin.bias.data = torch.tensor(b.copy())
        mods += [lin, nn.ReLU()]
        dims.append(W.shape[1])
    trunk = nn.Sequential(*mods)
    head_pi = nn.Linear(z["Wpi"].shape[0], 1)
    head_pi.weight.data = torch.tensor(z["Wpi"].T.copy())
    head_pi.bias.data = torch.tensor(z["bpi"].copy())
    head_v = nn.Linear(dims[-1], 1)   # fresh if the export has no value head
    if "Wv" in z.files:
        head_v.weight.data = torch.tensor(z["Wv"].T.copy())
        head_v.bias.data = torch.tensor(z["bv"].copy())
    emb = None
    if "Wemb" in z.files:            # v2: learned card-id embedding
        Wemb = z["Wemb"]
        emb = nn.Embedding(Wemb.shape[0], Wemb.shape[1], padding_idx=0)
        emb.weight.data = torch.tensor(Wemb.copy())
    norm = {k: z[k] for k in ("s_mu", "s_sd", "o_mu", "o_sd")}
    fv = int(z["feature_version"][0]) if "feature_version" in z.files else 1
    return trunk, head_pi, head_v, emb, norm, fv


def export(path, trunk, head_pi, head_v, emb, norm, fv):
    import torch.nn as nn
    w = {k: v.astype(np.float32) for k, v in norm.items()}
    w["feature_version"] = np.array([fv], dtype=np.int32)
    li = 0
    for m in trunk:
        if isinstance(m, nn.Linear):
            w[f"W{li}"] = m.weight.detach().numpy().T.astype(np.float32)
            w[f"b{li}"] = m.bias.detach().numpy().astype(np.float32)
            li += 1
    w["Wpi"] = head_pi.weight.detach().numpy().T.astype(np.float32)
    w["bpi"] = head_pi.bias.detach().numpy().astype(np.float32)
    w["Wv"] = head_v.weight.detach().numpy().T.astype(np.float32)
    w["bv"] = head_v.bias.detach().numpy().astype(np.float32)
    if emb is not None:
        w["Wemb"] = emb.weight.detach().numpy().astype(np.float32)
    w["n_layers"] = np.array([li], dtype=np.int32)
    np.savez_compressed(path, **w)


# ───────────────────────────────── training ──────────────────────────────────

def rl_update(model_path, data, lr, epochs, batch, ent_coef, clip_kl):
    import torch
    import torch.nn as nn

    states, opts, ptr, ys, rew, ids = data
    trunk, head_pi, head_v, emb, norm, fv = load_torch(model_path)
    params = list(trunk.parameters()) + list(head_pi.parameters()) + list(head_v.parameters())
    if emb is not None:
        params += list(emb.parameters())
    optim = torch.optim.Adam(params, lr=lr)

    S = torch.from_numpy(np.clip((states - norm["s_mu"]) / norm["s_sd"], -10, 10).astype(np.float32))
    O = torch.from_numpy(np.clip((opts - norm["o_mu"]) / norm["o_sd"], -10, 10).astype(np.float32))
    IDS = torch.from_numpy(ids) if ids is not None else None
    PTR = torch.from_numpy(ptr)
    Y = torch.from_numpy(ys.astype(np.int64))
    R = torch.from_numpy(rew)
    n_dec = len(ys)

    stats = {}
    for ep in range(epochs):
        perm = np.random.permutation(n_dec)
        for i in range(0, n_dec, batch):
            dec = torch.from_numpy(perm[i:i + batch].astype(np.int64))
            counts = (PTR[dec + 1] - PTR[dec]).long()
            rows = torch.cat([torch.arange(PTR[j], PTR[j + 1]) for j in dec])
            seg = torch.repeat_interleave(torch.arange(len(dec)), counts)
            parts = [S[dec][seg], O[rows]]
            if emb is not None:
                parts.append(emb(IDS[rows]))
            x = torch.cat(parts, dim=1)

            h = trunk(x)
            logit = head_pi(h).squeeze(-1)

            first = torch.zeros(len(dec), dtype=torch.long)
            if len(dec) > 1:
                first[1:] = torch.cumsum(counts, 0)[:-1]
            pos = first + Y[dec]

            m = torch.full((len(dec),), -1e9).scatter_reduce(0, seg, logit, reduce="amax", include_self=True)
            ex = torch.exp(logit - m[seg])
            denom = torch.zeros(len(dec)).scatter_add(0, seg, ex)
            logp = (logit[pos] - m) - torch.log(denom + 1e-9)

            v = head_v(h[first]).squeeze(-1)
            adv = (R[dec] - v).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)   # normalise: big variance otherwise

            # entropy bonus keeps the policy from collapsing onto one option
            p = ex / denom[seg]
            ent = -(p * torch.log(p + 1e-9))
            ent = torch.zeros(len(dec)).scatter_add(0, seg, ent).mean()

            loss_pi = -(logp * adv).mean() - ent_coef * ent
            loss_v = ((v - R[dec]) ** 2).mean()
            loss = loss_pi + 0.5 * loss_v

            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
        stats = {"entropy": float(ent.detach()), "value_loss": float(loss_v.detach())}
    return trunk, head_pi, head_v, emb, norm, fv, stats


# ───────────────────────────────── evaluation ────────────────────────────────

def _wr(a0, a1, games):
    import io, contextlib
    from harness import run_match
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = run_match(a0, a1, games, verbose=False)
    w = r["wins"]
    d = w[0] + w[1]
    return w[0] / d if d else 0.5


def evaluate(model_path, games, baseline_path=None):
    """Two proxies, neither of which is ladder truth.

    vs_heur: CONTAMINATED -- the heuristic (and heuristic-piloted gauntlet decks) are
             in the training opponent pool, so we partly optimise against it directly.
             Reported only to catch catastrophic collapse.
    vs_bc:   improvement over our frozen starting policy. Less circular (the BC clone
             is not a training opponent), so this is the number to actually watch.
    """
    from decks.mega_lucario import DECK
    from agents.policy_agent import make_agent as pol
    from agents.heuristic import make_agent as heur
    vs_heur = _wr(pol(DECK, model_path=model_path), heur(DECK), games)
    vs_bc = None
    if baseline_path:
        vs_bc = _wr(pol(DECK, model_path=model_path),
                    pol(DECK, model_path=baseline_path), games)
    return vs_heur, vs_bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="agents/policy.npz")
    ap.add_argument("--outdir", default="ckpt")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--eval-games", type=int, default=60)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cur = os.path.abspath(args.model)
    frozen_bc = cur   # never changes: the yardstick we must beat
    base_heur, _ = evaluate(cur, args.eval_games)
    print(f"[iter 0] BC baseline: vs_heuristic={base_heur:.1%} (frozen BC = yardstick)")
    best_bc, best_path = 0.5, cur

    for it in range(1, args.iters + 1):
        t0 = time.perf_counter()
        data = generate(cur, args.games, args.workers, args.temp, seed=it * 1000)
        n_dec = len(data[3])
        winrate_sp = float((data[4] > 0).mean())
        t_gen = time.perf_counter() - t0

        t1 = time.perf_counter()
        trunk, hpi, hv, emb, norm, fv, st = rl_update(cur, data, args.lr, args.epochs,
                                                      args.batch, args.ent, None)
        path = os.path.join(args.outdir, f"policy_it{it}.npz")
        export(path, trunk, hpi, hv, emb, norm, fv)
        t_train = time.perf_counter() - t1

        vs_heur, vs_bc = evaluate(path, args.eval_games, baseline_path=frozen_bc)
        tag = ""
        if vs_bc is not None and vs_bc > best_bc:
            best_bc, best_path = vs_bc, path
            tag = "  <-- best"
        print(f"[iter {it}] dec={n_dec:6d} sp_win={winrate_sp:.1%} "
              f"ent={st['entropy']:.2f} vloss={st['value_loss']:.3f} | "
              f"vs_BC={vs_bc:.1%} vs_heur={vs_heur:.1%} "
              f"(gen {t_gen:.0f}s, train {t_train:.0f}s){tag}")
        cur = path

    print(f"\nbest: {best_path} at {best_bc:.1%} vs frozen BC")
    print("vs_BC is the signal to trust (BC is not a training opponent).")
    print("vs_heur is CONTAMINATED: the heuristic is in the training pool.")
    print("Neither is ladder truth -- only the ladder is.")


if __name__ == "__main__":
    main()
