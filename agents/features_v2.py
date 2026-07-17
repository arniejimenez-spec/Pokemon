"""Feature encoding v2: v1 + card identity per option.

Why v2 exists: v1 encodes cards by ATTRIBUTES only, which makes 10.7% of real
decisions contain literally indistinguishable options (all Items share identical
attribute vectors: Ultra Ball == Night Stretcher == Switch). That blindness matches
the ~87% BC agreement ceiling, and a 7x bigger net (130k -> 916k params) moved
agreement only +0.2% -- the input was the constraint, not capacity.

v2 keeps the v1 state/option features unchanged and additionally returns the raw
card id per option. The network embeds ids (learned 16-dim table over 1268 ids) and
concatenates the embedding to the option features. Unknown/absent ids map to 0.

This is a SEPARATE module (not an in-place change): shipped v1 models must remain
rebuildable, so `policy_agent` dispatches on the feature_version stored in the
weights file.
"""
import numpy as np

try:
    from agents import features as _v1
    from agents.heuristic import resolve_card_id
except ImportError:
    import features as _v1
    from heuristic import resolve_card_id

FEATURE_VERSION = 2
N_CARD_IDS = 1268          # engine ids are 1..1267; 0 = unknown/none
EMB_DIM = 16               # fixed by the trainer; recorded here for reference

STATE_DIM = _v1.STATE_DIM
OPTION_DIM = _v1.OPTION_DIM
encode_state = _v1.encode_state
encode_options = _v1.encode_options


def encode_option_ids(obs) -> np.ndarray:
    """Raw card id per option (0 when the option has no resolvable card)."""
    ids = []
    for opt in obs.select.option:
        cid = resolve_card_id(obs, opt)
        if cid is None or not (0 < cid < N_CARD_IDS):
            cid = 0
        ids.append(cid)
    return np.array(ids, dtype=np.int32)


def encode(obs):
    """Return (state_vec[STATE_DIM], option_mat[n, OPTION_DIM], option_ids[n])."""
    return _v1.encode_state(obs), _v1.encode_options(obs), encode_option_ids(obs)
