"""Kaggle entrypoint: Pokemon TCG AI Battle Challenge — determinized-rollout search agent."""
import os

from search_agent import make_agent


def read_deck_csv() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path) as f:
        lines = f.read().split("\n")
    return [int(lines[i]) for i in range(60)]


_agent = make_agent(read_deck_csv())


def agent(obs_dict: dict) -> list[int]:
    return _agent(obs_dict)
