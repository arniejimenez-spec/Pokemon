"""Generate a behavioural-cloning dataset from heuristic self-play.

Records, for every non-forced decision, the encoded state + the encoded options the
engine offered + which option the heuristic picked (the label) + the eventual game
outcome for the deciding player (kept for later value-head / RL work).

Exploration (DAgger-style): with probability `epsilon` we *play* a random legal
option but still record the *heuristic's* choice as the label. That drives the game
into states greedy play never reaches, while the label stays expert — so the clone
learns to recover from mistakes instead of only seeing the on-policy path.

Storage is ragged: states are stored once per decision (173 floats) and option rows
once per option (74 floats), joined via `opt_ptr`. Concatenating state onto every
option row would cost ~3x the memory.

Usage:
    python train/gen_bc_data.py --games 2000 --workers 8 --out data/
"""
import argparse
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# v3's four state-level bags, in a fixed order shared by the worker, the
# merge step, and the saved npz key names.
BAG_NAMES = ("hand", "deckrem", "owndisc", "oppdisc")


class RaggedList:
    """Accumulates variable-length int arrays, one per decision, with a ptr
    array marking decision boundaries -- the same ragged-storage pattern
    already used for `opts`/`opt_ptr`, reused here for the four v3 bags so we
    don't hand-roll the same ptr bookkeeping four more times."""

    def __init__(self):
        self.chunks = []
        self.ptr = [0]

    def add(self, arr):
        arr = np.asarray(arr, dtype=np.int32)
        self.chunks.append(arr)
        self.ptr.append(self.ptr[-1] + len(arr))

    def arrays(self):
        ids = np.concatenate(self.chunks) if self.chunks else np.array([], dtype=np.int32)
        return ids, np.array(self.ptr, dtype=np.int64)


def _play_games(args):
    """Worker: play `n_games` and return arrays for the decisions seen."""
    n_games, seed, opp_name, epsilon, fv = args

    from decks.mega_lucario import DECK as LUCARIO
    from decks.gauntlet import GAUNTLET
    from agents.heuristic import HeuristicPolicy
    if fv == 3:
        from agents import features_v3 as F
    elif fv == 2:
        from agents import features_v2 as F
    else:
        from agents import features as F
    from cg.game import battle_start, battle_select, battle_finish
    from cg.api import to_observation_class

    rng = random.Random(seed)
    opp_deck = LUCARIO if opp_name == "mirror" else GAUNTLET[opp_name]
    decks = (LUCARIO, opp_deck)
    pols = (HeuristicPolicy(LUCARIO), HeuristicPolicy(opp_deck))

    states, opts, ptr, ys, seats, turns = [], [], [0], [], [], []
    opt_ids = []
    bags = {name: RaggedList() for name in BAG_NAMES}
    game_winner = []

    for g in range(n_games):
        obs, sd = battle_start(decks[0], decks[1])
        if obs is None:
            continue
        g_start = len(ys)
        steps = 0
        while obs["current"]["result"] == -1:
            o = to_observation_class(obs)
            sel = o.select
            seat = o.current.yourIndex
            n_opt = len(sel.option)

            # record only real decisions: pick-one among several options
            recordable = (n_opt > 1 and sel.maxCount == 1 and sel.minCount == 1)
            choice = pols[seat].select(o)

            if recordable:
                enc = F.encode(o, decks[seat]) if fv == 3 else F.encode(o)
                states.append(enc[0])
                opts.append(enc[1])
                if fv >= 2:
                    opt_ids.append(enc[2])
                if fv == 3:
                    for name, arr in zip(BAG_NAMES, enc[3:7]):
                        bags[name].add(arr)
                ptr.append(ptr[-1] + n_opt)
                ys.append(choice[0])
                seats.append(seat)
                turns.append(min(o.current.turn, 40))

            # exploration: play a random legal option, but the label above stays expert
            if epsilon > 0 and n_opt > 1 and rng.random() < epsilon:
                k = rng.randint(sel.minCount, sel.maxCount)
                play = rng.sample(range(n_opt), k) if k > 0 else []
            else:
                play = choice

            obs = battle_select(play)
            steps += 1
            if steps > 3000:
                break

        result = obs["current"]["result"]
        battle_finish()
        for _ in range(len(ys) - g_start):
            game_winner.append(result)

    if not ys:
        return None
    bag_arrays = {name: bags[name].arrays() for name in BAG_NAMES} if fv == 3 else None
    return (
        np.stack(states).astype(np.float32),
        np.concatenate(opts).astype(np.float32),
        np.array(ptr, dtype=np.int64),
        np.array(ys, dtype=np.int16),
        np.array(seats, dtype=np.int8),
        np.array(turns, dtype=np.int16),
        np.array(game_winner, dtype=np.int8),
        (np.concatenate(opt_ids).astype(np.int16) if fv >= 2 else None),
        bag_arrays,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=2000, help="total games across all workers")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--epsilon", type=float, default=0.15, help="exploration rate (labels stay expert)")
    p.add_argument("--fv", type=int, default=1, choices=(1, 2, 3), help="feature version")
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # spread games over a mix of opponents so the net sees diverse boards
    opponents = ["mirror", "zacian", "yveltal", "latias"]
    jobs = []
    per_worker = max(1, args.games // args.workers)
    for w in range(args.workers):
        jobs.append((per_worker, args.seed + w * 977, opponents[w % len(opponents)],
                     args.epsilon, args.fv))

    t0 = time.perf_counter()
    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        results = [r for r in pool.map(_play_games, jobs) if r is not None]
    dt = time.perf_counter() - t0

    if not results:
        raise SystemExit("no data generated")

    # merge shards, fixing up the ragged option pointers
    states = np.concatenate([r[0] for r in results])
    ys = np.concatenate([r[3] for r in results])
    seats = np.concatenate([r[4] for r in results])
    turns = np.concatenate([r[5] for r in results])
    wins = np.concatenate([r[6] for r in results])
    opt_list, ptr_list, off = [], [0], 0
    for r in results:
        opt_list.append(r[1])
        ptr_list.extend((r[2][1:] + off).tolist())
        off += r[1].shape[0]
    opts = np.concatenate(opt_list)
    ptr = np.array(ptr_list, dtype=np.int64)

    # outcome from the deciding player's perspective: +1 win, -1 loss, 0 draw
    outcome = np.where(wins == 2, 0, np.where(wins == seats, 1, -1)).astype(np.int8)

    extra = {}
    if args.fv >= 2:
        extra["opt_ids"] = np.concatenate([r[7] for r in results])
        assert extra["opt_ids"].shape[0] == opts.shape[0], "opt_ids misaligned"

    if args.fv == 3:
        # each of the 4 bags: concat ids across shards, re-base each shard's ptr
        # by the running id-offset (same pattern as opt_ptr above)
        for name in BAG_NAMES:
            id_list, ptr_list, off = [], [0], 0
            for r in results:
                ids, ptr_r = r[8][name]
                id_list.append(ids)
                ptr_list.extend((ptr_r[1:] + off).tolist())
                off += ids.shape[0]
            extra[f"{name}_ids"] = np.concatenate(id_list) if id_list else np.array([], dtype=np.int32)
            extra[f"{name}_ptr"] = np.array(ptr_list, dtype=np.int64)
            assert len(extra[f"{name}_ptr"]) == len(ys) + 1, f"{name}_ptr misaligned"

    names = {1: "bc_data.npz", 2: "bc_data_v2.npz", 3: "bc_data_v3.npz"}
    path = os.path.join(args.out, names[args.fv])
    np.savez_compressed(path, states=states, opts=opts, opt_ptr=ptr, y=ys,
                        seat=seats, turn=turns, outcome=outcome,
                        feature_version=np.array([args.fv], dtype=np.int32), **extra)
    mb = os.path.getsize(path) / 1e6
    print(f"games={args.games} workers={args.workers} in {dt:.0f}s ({args.games/dt:.1f} games/s)")
    print(f"decisions={len(ys)}  options={opts.shape[0]}  avg_opts={opts.shape[0]/len(ys):.1f}")
    print(f"outcome balance: win={np.mean(outcome==1):.1%} loss={np.mean(outcome==-1):.1%} draw={np.mean(outcome==0):.1%}")
    print(f"wrote {path} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
