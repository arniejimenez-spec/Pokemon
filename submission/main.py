"""Kaggle entrypoint: Pokemon TCG AI Battle Challenge.

Currently ships the plain heuristic policy. The determinized search agent
(search_agent.py) scored *worse* on the live ladder (456 vs 601) despite winning
67-75% against this heuristic locally -- because every local search evaluation
used this same heuristic as the opponent, which is exactly the opponent policy the
search assumes in its rollouts. Do not ship search again without an evaluation
against opponents that are not this heuristic.
"""
import os

from heuristic import make_agent


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
