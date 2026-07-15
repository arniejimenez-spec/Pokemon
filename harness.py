"""Local evaluation harness: pit two competition-style agents against each other.

An agent is a callable `agent(obs_dict: dict) -> list[int]`, exactly as on Kaggle:
the first call has obs["select"] == None and must return the 60-card deck.

Usage:
    python harness.py --games 20
"""
import argparse
import importlib
import time

from cg.game import battle_start, battle_select, battle_finish

DECK_REQUEST = {"select": None, "logs": [], "current": None}
MAX_STEPS = 50000


def play_game(agent0, agent1, swap: bool = False) -> dict:
    """Play one game. If swap, agent1 is seated as player 0.

    Returns dict with keys: winner (0/1 index of *agent*, 2=draw), steps, time0, time1.
    """
    agents = (agent1, agent0) if swap else (agent0, agent1)
    deck0 = agents[0](DECK_REQUEST)
    deck1 = agents[1](DECK_REQUEST)

    obs_dict, start_data = battle_start(deck0, deck1)
    if obs_dict is None:
        raise RuntimeError(
            f"battle_start failed: errorPlayer={start_data.errorPlayer} "
            f"errorType={start_data.errorType} (bad deck?)")

    times = [0.0, 0.0]
    steps = 0
    try:
        while True:
            result = obs_dict["current"]["result"]
            if result != -1:
                break
            seat = obs_dict["current"]["yourIndex"]
            t0 = time.perf_counter()
            choice = agents[seat](obs_dict)
            times[seat] += time.perf_counter() - t0
            obs_dict = battle_select(choice)
            steps += 1
            if steps > MAX_STEPS:
                raise RuntimeError("game exceeded MAX_STEPS")
    finally:
        battle_finish()

    if result == 2:
        winner = 2
    else:
        winner = result if not swap else 1 - result
    a_time, b_time = (times[1], times[0]) if swap else (times[0], times[1])
    return {"winner": winner, "steps": steps, "time0": a_time, "time1": b_time}


def run_match(agent0, agent1, games: int, verbose: bool = True) -> dict:
    wins = [0, 0, 0]  # agent0, agent1, draws
    total_time = [0.0, 0.0]
    for g in range(games):
        r = play_game(agent0, agent1, swap=(g % 2 == 1))
        wins[r["winner"]] += 1
        total_time[0] += r["time0"]
        total_time[1] += r["time1"]
        if verbose:
            print(f"  game {g+1}/{games}: winner={'draw' if r['winner']==2 else 'agent'+str(r['winner'])} "
                  f"steps={r['steps']}")
    decided = wins[0] + wins[1]
    wr = wins[0] / decided if decided else 0.5
    print(f"agent0 {wins[0]} — agent1 {wins[1]} — draws {wins[2]}  "
          f"(agent0 win rate {wr:.1%} of decided)")
    print(f"avg think time/game: agent0 {total_time[0]/games:.2f}s, agent1 {total_time[1]/games:.2f}s")
    return {"wins": wins, "win_rate": wr}


def load_agent(spec: str):
    """Load 'module:factory_args' — module must expose make_agent(deck) and DECK, or agent()."""
    mod = importlib.import_module(spec)
    if hasattr(mod, "agent"):
        return mod.agent
    deck = getattr(mod, "DECK")
    return mod.make_agent(deck)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a0", default="agents.random_agent")
    p.add_argument("--a1", default="agents.random_agent")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--deck", default="sample_submission/sample_submission/deck.csv",
                   help="deck.csv used for agents loaded via make_agent")
    args = p.parse_args()

    with open(args.deck) as f:
        deck = [int(x) for x in f.read().split("\n")[:60]]

    def load(spec):
        mod = importlib.import_module(spec)
        if hasattr(mod, "agent"):
            return mod.agent
        return mod.make_agent(deck)

    run_match(load(args.a0), load(args.a1), args.games)


if __name__ == "__main__":
    main()
