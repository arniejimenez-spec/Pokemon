"""Smoke test: random vs random battle using the local cabt engine."""
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_submission", "sample_submission"))

from cg.api import to_observation_class, all_card_data
from cg.game import battle_start, battle_select, battle_finish


def read_deck() -> list[int]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sample_submission", "sample_submission", "deck.csv")
    with open(path) as f:
        lines = f.read().split("\n")
    return [int(lines[i]) for i in range(60)]


def main():
    random.seed(42)
    deck = read_deck()
    cards = all_card_data()
    print(f"Engine loaded OK. Total cards in pool: {len(cards)}")

    obs_dict, start_data = battle_start(deck, deck)
    if obs_dict is None:
        print(f"Battle failed to start: errorPlayer={start_data.errorPlayer} errorType={start_data.errorType}")
        return 1

    steps = 0
    while True:
        obs = to_observation_class(obs_dict)
        if obs.current is not None and obs.current.result != -1:
            print(f"Game over after {steps} selections. Result: player {obs.current.result} "
                  f"({'draw' if obs.current.result == 2 else 'win'})")
            break
        sel = obs.select
        n = len(sel.option)
        count = random.randint(sel.minCount, sel.maxCount)
        choice = random.sample(range(n), count)
        obs_dict = battle_select(choice)
        steps += 1
        if steps > 20000:
            print("Aborting: too many steps")
            break

    battle_finish()
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
