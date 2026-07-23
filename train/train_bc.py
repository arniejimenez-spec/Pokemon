"""Train the option-scoring policy by behavioural cloning, export numpy weights.

Architecture
------------
The action set is variable-length, so there is no fixed policy head. Instead each
option is scored independently and we softmax over the options actually offered:

    logit_i = MLP([state, option_i, ...])     # shared MLP, applied per option
    loss    = cross_entropy(softmax_i(logit), expert_choice)

Ragged batches are handled by flattening every option in the batch into one matrix,
running a single forward pass, then segment-softmaxing back per decision. That keeps
the GPU busy with one big matmul instead of per-decision loops.

Feature versions (--fv, must match the --data file's own feature_version):
  1: state + option features only.
  2: + a learned per-option card-identity embedding (fixes v1's blindness where
     e.g. all Items share identical attribute vectors).
  3: + four state-level id-bags (own hand / deck-remaining / own discard /
     opponent discard), each summed through a second learned embedding table --
     pooled at DECISION granularity (one bag per decision, not per option), via
     the same scatter-add trick used for the option-level segment-softmax.

Runs on CPU or GPU (use a Kaggle GPU notebook: this repo has no local GPU).
Exports a numpy weights file — the inference agent (agents/policy_agent.py) does a
pure-numpy forward pass so the submission never depends on torch being present.

Usage:
    python train/train_bc.py --data data/bc_data_v3.npz --fv 3 --out models/policy_bc_v3.npz
"""
import argparse
import os
import time

import numpy as np

EMB_DIM = 16   # must match agents.features_v2.EMB_DIM / agents.features_v3.STATE_EMB_DIM
N_CARD_IDS = 1268
BAG_NAMES = ("hand", "deckrem", "owndisc", "oppdisc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bc_data.npz")
    ap.add_argument("--out", default="models/policy_bc.npz")
    ap.add_argument("--fv", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512, help="decisions per batch")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    def pick_device():
        if not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            torch.zeros(8, device="cuda").relu_()   # some Kaggle GPUs (P100/sm_60)
            return torch.device("cuda")              # are incompatible with the
        except Exception as e:                       # preinstalled torch build
            print("GPU unusable, using CPU:", str(e)[:70])
            return torch.device("cpu")
    dev = pick_device()
    print(f"device: {dev}")

    d = np.load(args.data)
    fv_in_data = int(d["feature_version"][0]) if "feature_version" in d.files else 1
    assert fv_in_data == args.fv, f"--fv {args.fv} but data is feature_version {fv_in_data}"

    states, opts, ptr, y = d["states"], d["opts"], d["opt_ptr"], d["y"].astype(np.int64)
    outcome = d["outcome"].astype(np.float32)
    n_dec = len(y)
    print(f"decisions={n_dec} options={opts.shape[0]} state_dim={states.shape[1]} opt_dim={opts.shape[1]}")
    assert (y < np.diff(ptr)).all(), "label out of range -- bad dataset"

    idx = np.random.default_rng(0).permutation(n_dec)
    n_val = int(n_dec * args.val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    # Feature standardisation, stored in the npz so inference matches exactly.
    # Floor the std: many dims are one-hots that are near-constant in training, and
    # dividing by ~0 would blow an out-of-distribution value (an unseen context or
    # card attribute on the ladder) up to ~1e6 and wreck the logits. Clipping below
    # bounds it further. Embeddings are NOT standardised (they're learned params).
    s_mu, s_sd = states.mean(0), np.maximum(states.std(0), 1e-2)
    o_mu, o_sd = opts.mean(0), np.maximum(opts.std(0), 1e-2)

    opt_ids = torch.from_numpy(d["opt_ids"].astype(np.int64)) if args.fv >= 2 else None
    bag_data = None
    if args.fv == 3:
        bag_data = {name: (torch.from_numpy(np.clip(d[f"{name}_ids"], 0, N_CARD_IDS - 1).astype(np.int64)),
                            torch.from_numpy(d[f"{name}_ptr"]))
                    for name in BAG_NAMES}

    in_dim = states.shape[1] + opts.shape[1]
    if args.fv >= 2:
        in_dim += EMB_DIM
    if args.fv == 3:
        in_dim += 4 * EMB_DIM

    layers, prev = [], in_dim
    for _ in range(args.layers):
        layers += [nn.Linear(prev, args.hidden), nn.ReLU()]
        prev = args.hidden
    trunk = nn.Sequential(*layers).to(dev)
    head_pi = nn.Linear(prev, 1).to(dev)      # option logit
    head_v = nn.Linear(prev, 1).to(dev)       # state value (aux task, helps features)
    params = list(trunk.parameters()) + list(head_pi.parameters()) + list(head_v.parameters())

    emb = emb_state = None
    if args.fv >= 2:
        emb = nn.Embedding(N_CARD_IDS, EMB_DIM, padding_idx=0).to(dev)
        params += list(emb.parameters())
    if args.fv == 3:
        emb_state = nn.Embedding(N_CARD_IDS, EMB_DIM, padding_idx=0).to(dev)
        params += list(emb_state.parameters())

    optim = torch.optim.Adam(params, lr=args.lr)
    print(f"params: {sum(p.numel() for p in params):,}")

    S = torch.from_numpy(np.clip((states - s_mu) / s_sd, -10, 10).astype(np.float32))
    O = torch.from_numpy(np.clip((opts - o_mu) / o_sd, -10, 10).astype(np.float32))
    PTR = torch.from_numpy(ptr)
    Y = torch.from_numpy(y)
    OUT = torch.from_numpy(outcome)

    def pool_bag(dec_ids, ids_all, bag_ptr, table):
        """Sum-pool one bag's embeddings per decision (decision-granularity, not
        per-option) -- same scatter-add pattern as the option-level segment-softmax
        below, just applied to a differently-shaped ragged array."""
        counts = (bag_ptr[dec_ids + 1] - bag_ptr[dec_ids]).long()
        if counts.sum() == 0:
            return torch.zeros(len(dec_ids), EMB_DIM, device=dev)
        rows = torch.cat([torch.arange(bag_ptr[i], bag_ptr[i + 1]) for i in dec_ids])
        seg = torch.repeat_interleave(torch.arange(len(dec_ids)), counts).to(dev)
        vecs = table(ids_all[rows].to(dev))
        return torch.zeros(len(dec_ids), EMB_DIM, device=dev).scatter_add(
            0, seg.unsqueeze(-1).expand(-1, EMB_DIM), vecs)

    def run_batch(dec_ids, train: bool):
        # flatten every option of every decision in the batch into one matrix
        counts = (PTR[dec_ids + 1] - PTR[dec_ids]).long()
        rows = torch.cat([torch.arange(PTR[i], PTR[i + 1]) for i in dec_ids])
        seg = torch.repeat_interleave(torch.arange(len(dec_ids)), counts)
        parts = [S[dec_ids][seg].to(dev), O[rows].to(dev)]
        if args.fv >= 2:
            parts.append(emb(opt_ids[rows].to(dev)))
        seg = seg.to(dev)

        if args.fv == 3:
            bag_vecs = [pool_bag(dec_ids, bag_data[name][0], bag_data[name][1], emb_state)
                        for name in BAG_NAMES]
            # each bag vector is per-DECISION; broadcast to every option row of
            # that decision via the same seg index used for the base state
            for bv in bag_vecs:
                parts.append(bv[seg])

        x = torch.cat(parts, dim=1)
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
            optim.zero_grad(); loss.backward(); optim.step()

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
         "feature_version": np.array([args.fv], dtype=np.int32)}
    li = 0
    for m in trunk:
        if isinstance(m, nn.Linear):
            w[f"W{li}"] = m.weight.detach().cpu().numpy().T.astype(np.float32)
            w[f"b{li}"] = m.bias.detach().cpu().numpy().astype(np.float32)
            li += 1
    w["Wpi"] = head_pi.weight.detach().cpu().numpy().T.astype(np.float32)
    w["bpi"] = head_pi.bias.detach().cpu().numpy().astype(np.float32)
    w["Wv"] = head_v.weight.detach().cpu().numpy().T.astype(np.float32)
    w["bv"] = head_v.bias.detach().cpu().numpy().astype(np.float32)
    if emb is not None:
        w["Wemb"] = emb.weight.detach().cpu().numpy().astype(np.float32)
    if emb_state is not None:
        w["Wemb_state"] = emb_state.weight.detach().cpu().numpy().astype(np.float32)
    w["n_layers"] = np.array([li], dtype=np.int32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **w)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
