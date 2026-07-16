"""Train the option-scoring policy by behavioural cloning, export numpy weights.

Architecture
------------
The action set is variable-length, so there is no fixed policy head. Instead each
option is scored independently and we softmax over the options actually offered:

    logit_i = MLP([state, option_i])          # shared MLP, applied per option
    loss    = cross_entropy(softmax_i(logit), expert_choice)

Ragged batches are handled by flattening every option in the batch into one matrix,
running a single forward pass, then segment-softmaxing back per decision. That keeps
the GPU busy with one big matmul instead of per-decision loops.

Runs on CPU or GPU (use a Kaggle GPU notebook: this repo has no local GPU).
Exports `policy.npz` — the inference agent does a pure-numpy forward pass so the
submission never depends on torch being present.

Usage:
    python train/train_bc.py --data data/bc_data.npz --out submission/policy.npz --epochs 12
"""
import argparse
import os
import time

import numpy as np


def segment_softmax_ce(logits, ptr, y, n_dec):
    """Cross-entropy over ragged groups. Pure numpy reference (torch version below)."""
    loss = 0.0
    for d in range(n_dec):
        lo, hi = ptr[d], ptr[d + 1]
        z = logits[lo:hi]
        z = z - z.max()
        p = np.exp(z) / np.exp(z).sum()
        loss -= np.log(max(p[y[d]], 1e-9))
    return loss / n_dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bc_data.npz")
    ap.add_argument("--out", default="submission/policy.npz")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512, help="decisions per batch")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    d = np.load(args.data)
    states, opts, ptr, y = d["states"], d["opts"], d["opt_ptr"], d["y"].astype(np.int64)
    outcome = d["outcome"].astype(np.float32)
    n_dec = len(y)
    print(f"decisions={n_dec} options={opts.shape[0]} state_dim={states.shape[1]} opt_dim={opts.shape[1]}")

    # split by decision (val is a held-out slice, not shuffled rows within a group)
    idx = np.random.default_rng(0).permutation(n_dec)
    n_val = int(n_dec * args.val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    # Feature standardisation, stored in the npz so inference matches exactly.
    # Floor the std: many dims are one-hots that are near-constant in training, and
    # dividing by ~0 would blow an out-of-distribution value (an unseen context or
    # card attribute on the ladder) up to ~1e6 and wreck the logits. Clipping below
    # bounds it further. Inference MUST apply the identical floor + clip.
    s_mu, s_sd = states.mean(0), np.maximum(states.std(0), 1e-2)
    o_mu, o_sd = opts.mean(0), np.maximum(opts.std(0), 1e-2)

    in_dim = states.shape[1] + opts.shape[1]
    layers, prev = [], in_dim
    for _ in range(args.layers):
        layers += [nn.Linear(prev, args.hidden), nn.ReLU()]
        prev = args.hidden
    trunk = nn.Sequential(*layers).to(dev)
    head_pi = nn.Linear(prev, 1).to(dev)      # option logit
    head_v = nn.Linear(prev, 1).to(dev)       # state value (aux task, helps features)
    params = list(trunk.parameters()) + list(head_pi.parameters()) + list(head_v.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    print(f"params: {sum(p.numel() for p in params):,}")

    S = torch.from_numpy(np.clip((states - s_mu) / s_sd, -10, 10).astype(np.float32))
    O = torch.from_numpy(np.clip((opts - o_mu) / o_sd, -10, 10).astype(np.float32))
    PTR = torch.from_numpy(ptr)
    Y = torch.from_numpy(y)
    OUT = torch.from_numpy(outcome)

    def run_batch(dec_ids, train: bool):
        # flatten every option of every decision in the batch into one matrix
        counts = (PTR[dec_ids + 1] - PTR[dec_ids]).long()
        rows = torch.cat([torch.arange(PTR[i], PTR[i + 1]) for i in dec_ids])
        seg = torch.repeat_interleave(torch.arange(len(dec_ids)), counts)
        x = torch.cat([S[dec_ids][seg], O[rows]], dim=1).to(dev)
        seg = seg.to(dev)

        h = trunk(x)
        logit = head_pi(h).squeeze(-1)

        # Row where each decision's option block starts in the flattened batch.
        # dec_ids is shuffled, so `rows` is NOT globally sorted -- never searchsorted
        # it. Blocks are laid out consecutively, so block_start + y is the expert row.
        first = torch.zeros(len(dec_ids), dtype=torch.long)
        if len(dec_ids) > 1:
            first[1:] = torch.cumsum(counts, 0)[:-1]
        first = first.to(dev)
        pos = first + Y[dec_ids].to(dev)

        # segment softmax -> cross entropy against the expert's option
        m = torch.full((len(dec_ids),), -1e9, device=dev).scatter_reduce(
            0, seg, logit, reduce="amax", include_self=True)
        ex = torch.exp(logit - m[seg])
        denom = torch.zeros(len(dec_ids), device=dev).scatter_add(0, seg, ex)
        logp = (logit[pos] - m) - torch.log(denom + 1e-9)
        loss_pi = -logp.mean()

        # aux value head, read off each decision's first row (state-level)
        v = head_v(h[first]).squeeze(-1)
        loss_v = ((v - OUT[dec_ids].to(dev)) ** 2).mean()

        loss = loss_pi + 0.5 * loss_v
        if train:
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            # agreement: is the expert's option the argmax of its group?
            correct = (torch.abs(logit[pos] - m) < 1e-6).float().mean()
        return loss_pi.item(), loss_v.item(), correct.item()

    for ep in range(args.epochs):
        t0 = time.perf_counter()
        perm = np.random.permutation(tr_idx)
        tot = n = 0.0
        for i in range(0, len(perm), args.batch):
            b = torch.from_numpy(perm[i:i + args.batch].astype(np.int64))
            lp, lv, acc = run_batch(b, True)
            tot += lp; n += 1
        # validation
        with torch.no_grad():
            vb = torch.from_numpy(val_idx.astype(np.int64))
            accs = []
            for i in range(0, len(vb), args.batch):
                _, _, a = run_batch(vb[i:i + args.batch], False)
                accs.append(a)
        print(f"epoch {ep+1}/{args.epochs} train_loss={tot/max(n,1):.4f} "
              f"val_agree={np.mean(accs):.1%} ({time.perf_counter()-t0:.0f}s)")

    # export plain numpy weights for dependency-free inference
    w = {"s_mu": s_mu.astype(np.float32), "s_sd": s_sd.astype(np.float32),
         "o_mu": o_mu.astype(np.float32), "o_sd": o_sd.astype(np.float32),
         "feature_version": np.array([1], dtype=np.int32)}
    li = 0
    for m in trunk:
        if isinstance(m, nn.Linear):
            w[f"W{li}"] = m.weight.detach().cpu().numpy().T.astype(np.float32)
            w[f"b{li}"] = m.bias.detach().cpu().numpy().astype(np.float32)
            li += 1
    w["Wpi"] = head_pi.weight.detach().cpu().numpy().T.astype(np.float32)
    w["bpi"] = head_pi.bias.detach().cpu().numpy().astype(np.float32)
    w["n_layers"] = np.array([li], dtype=np.int32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **w)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
