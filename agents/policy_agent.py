"""Learned-policy agent: pure-numpy inference over the option-scoring network.

Loads weights exported by train/train_bc.py and scores each legal option:

    logit_i = MLP([state, option_i]);  choose argmax over the options offered

Because the network only ever scores options the engine handed us, illegal moves are
structurally impossible. Falls back to the heuristic whenever the model is missing or
anything goes wrong -- a crash forfeits the episode on the ladder.

No torch: the submission bundle must not depend on the Kaggle runtime's ML stack.
"""
import os

import numpy as np

from cg.api import Observation, to_observation_class

try:
    from agents.heuristic import HeuristicPolicy
    from agents import features as F
except ImportError:
    from heuristic import HeuristicPolicy
    import features as F


class PolicyNet:
    """Minimal MLP forward pass in numpy."""

    def __init__(self, path: str):
        z = np.load(path)
        self.s_mu, self.s_sd = z["s_mu"], z["s_sd"]
        self.o_mu, self.o_sd = z["o_mu"], z["o_sd"]
        n = int(z["n_layers"][0])
        self.W = [z[f"W{i}"] for i in range(n)]
        self.b = [z[f"b{i}"] for i in range(n)]
        self.Wpi, self.bpi = z["Wpi"], z["bpi"]
        fv = int(z["feature_version"][0]) if "feature_version" in z else -1
        if fv != F.FEATURE_VERSION:
            raise ValueError(
                f"feature_version mismatch: weights={fv} code={F.FEATURE_VERSION}")

    def logits(self, state: np.ndarray, opts: np.ndarray) -> np.ndarray:
        # Clip must match train_bc.py exactly: out-of-distribution values (unseen
        # contexts/cards on the ladder) would otherwise explode through the
        # near-constant one-hot dims and wreck the logits.
        s = np.clip((state - self.s_mu) / self.s_sd, -10, 10)
        o = np.clip((opts - self.o_mu) / self.o_sd, -10, 10)
        x = np.concatenate([np.repeat(s[None, :], o.shape[0], axis=0), o], axis=1)
        for W, b in zip(self.W, self.b):
            x = np.maximum(x @ W + b, 0.0)   # relu
        return (x @ self.Wpi + self.bpi).ravel()


def _find_model(filename: str = "policy.npz") -> str | None:
    for p in (filename,
              os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
              "/kaggle_simulations/agent/" + filename):
        if os.path.exists(p):
            return p
    return None


class PolicyAgent:
    def __init__(self, deck: list[int], model_path: str | None = None,
                 temperature: float = 0.0):
        self.deck = deck
        self.fallback = HeuristicPolicy(deck)
        self.temperature = temperature
        self.net = None
        path = model_path or _find_model()
        if path:
            try:
                self.net = PolicyNet(path)
            except Exception:
                self.net = None  # bad/missing weights -> heuristic

    def select(self, obs: Observation) -> list[int]:
        sel = obs.select
        opts = sel.option
        n = len(opts)
        lo, hi = sel.minCount, sel.maxCount

        # forced / degenerate: no decision to make
        if n == 0:
            return []
        if lo == hi == n:
            return list(range(n))
        if n == 1 and lo <= 1 <= hi:
            return [0]
        if self.net is None or obs.current is None:
            return self.fallback.select(obs)

        try:
            state, om = F.encode(obs)
            z = self.net.logits(state, om)
            if not np.all(np.isfinite(z)):
                return self.fallback.select(obs)
            k = hi if hi == lo else max(lo, 1)
            k = min(k, n)
            # NOTE: must be plain python ints -- the engine rejects np.int64
            # ("select_list is not list[int]"), which would error every decision.
            if self.temperature > 0:
                p = np.exp((z - z.max()) / self.temperature)
                p /= p.sum()
                return [int(i) for i in np.random.choice(n, size=k, replace=False, p=p)]
            return [int(i) for i in np.argsort(-z)[:k]]
        except Exception:
            return self.fallback.select(obs)


def make_agent(deck: list[int], **kw):
    ag = PolicyAgent(deck, **kw)

    def agent(obs_dict: dict) -> list[int]:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return deck
        try:
            return ag.select(obs)
        except Exception:
            sel = obs.select
            lo = sel.minCount
            return list(range(lo)) if lo > 0 else [0]

    return agent
