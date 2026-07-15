"""Uniform random agent — the floor for evaluation."""
import random

from cg.api import Observation, to_observation_class


def make_agent(deck: list[int]):
    def agent(obs_dict: dict) -> list[int]:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        n = len(obs.select.option)
        count = random.randint(obs.select.minCount, obs.select.maxCount)
        return random.sample(range(n), count)
    return agent
