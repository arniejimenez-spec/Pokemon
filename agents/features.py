"""Feature encoding for the learned policy.

Design notes
------------
The engine hands us a *variable-length* list of legal options per decision, across
~49 selection contexts. There is no fixed action space, so the policy scores each
option independently:

    logit_i = MLP([state_features, option_features_i])
    policy  = softmax over the options actually offered

That makes illegal actions structurally impossible and handles any option count.

Cards are encoded by **attributes** (type/HP/stage/ex/retreat/attack damage), not by
a card-id embedding. The ladder is full of decks we don't own, so the encoder must
generalize to cards never seen in training.

This module is imported by BOTH training and inference and must stay deterministic
and numpy-only. Any change here invalidates existing model weights.
"""
import numpy as np

from cg.api import (
    Observation, AreaType, CardType, EnergyType, SelectType, OptionType,
)

try:
    from agents.heuristic import CARD_DB, ATTACK_DB, resolve_card_id
except ImportError:
    from heuristic import CARD_DB, ATTACK_DB, resolve_card_id

FEATURE_VERSION = 1

# enum sizes (fixed upper bounds; new enum members may be appended by the engine)
N_SELECT_TYPE = 12
N_CONTEXT = 52
N_OPTION_TYPE = 18
N_CARD_TYPE = 8
N_ENERGY_TYPE = 12
N_AREA = 13

_MAX_HP = 400.0
_MAX_DMG = 350.0


def _onehot(idx, n):
    v = np.zeros(n, dtype=np.float32)
    if idx is not None and 0 <= int(idx) < n:
        v[int(idx)] = 1.0
    return v


def _best_damage(card_id):
    c = CARD_DB.get(card_id)
    if not c or not c.attacks:
        return 0.0
    return float(max((ATTACK_DB[a].damage for a in c.attacks if a in ATTACK_DB), default=0))


def _card_attrs(card_id) -> np.ndarray:
    """Attribute vector for a card id (zeros if unknown/None)."""
    c = CARD_DB.get(card_id) if card_id is not None else None
    if c is None:
        return np.zeros(N_CARD_TYPE + N_ENERGY_TYPE + 8, dtype=np.float32)
    return np.concatenate([
        _onehot(c.cardType, N_CARD_TYPE),
        _onehot(c.energyType, N_ENERGY_TYPE),
        np.array([
            (c.hp or 0) / _MAX_HP,
            1.0 if c.basic else 0.0,
            1.0 if c.stage1 else 0.0,
            1.0 if c.stage2 else 0.0,
            1.0 if c.ex else 0.0,
            1.0 if c.megaEx else 0.0,
            (c.retreatCost or 0) / 4.0,
            _best_damage(card_id) / _MAX_DMG,
        ], dtype=np.float32),
    ])


_CARD_ATTR_DIM = N_CARD_TYPE + N_ENERGY_TYPE + 8


def _mon_feats(mon) -> np.ndarray:
    """Live stats for a Pokemon in play (zeros if empty/facedown)."""
    if mon is None:
        return np.zeros(6 + _CARD_ATTR_DIM, dtype=np.float32)
    hp = mon.hp or 0
    mx = mon.maxHp or 1
    return np.concatenate([
        np.array([
            1.0,
            hp / _MAX_HP,
            hp / max(mx, 1),
            len(mon.energies or []) / 5.0,
            len(mon.tools or []) / 2.0,
            1.0 if mon.appearThisTurn else 0.0,
        ], dtype=np.float32),
        _card_attrs(mon.id),
    ])


_MON_DIM = 6 + _CARD_ATTR_DIM


def _side_feats(p) -> np.ndarray:
    bench_hp = sum((m.hp or 0) for m in p.bench if m)
    bench_e = sum(len(m.energies or []) for m in p.bench if m)
    return np.array([
        len(p.prize) / 6.0,
        (p.deckCount or 0) / 60.0,
        (p.handCount or 0) / 10.0,
        len(p.discard) / 60.0,
        len(p.bench) / 5.0,
        (p.benchMax or 5) / 5.0,
        bench_hp / 1000.0,
        bench_e / 10.0,
        1.0 if p.poisoned else 0.0,
        1.0 if p.burned else 0.0,
        1.0 if p.asleep else 0.0,
        1.0 if p.paralyzed else 0.0,
        1.0 if p.confused else 0.0,
    ], dtype=np.float32)


_SIDE_DIM = 13


def _weakness_pair(my_mon, opp_mon) -> np.ndarray:
    """Type-matchup flags between the two active Pokemon."""
    out = np.zeros(2, dtype=np.float32)
    if my_mon is None or opp_mon is None:
        return out
    mc, oc = CARD_DB.get(my_mon.id), CARD_DB.get(opp_mon.id)
    if mc and oc:
        if oc.weakness is not None and oc.weakness == mc.energyType:
            out[0] = 1.0   # we hit them for weakness
        if mc.weakness is not None and mc.weakness == oc.energyType:
            out[1] = 1.0   # they hit us for weakness
    return out


def encode_state(obs: Observation) -> np.ndarray:
    st = obs.current
    sel = obs.select
    me = st.yourIndex
    mp, op = st.players[me], st.players[1 - me]
    my_act = mp.active[0] if mp.active else None
    op_act = op.active[0] if op.active else None

    scalars = np.array([
        min(st.turn, 40) / 40.0,
        min(st.turnActionCount, 20) / 20.0,
        1.0 if st.firstPlayer == me else 0.0,
        1.0 if st.supporterPlayed else 0.0,
        1.0 if st.stadiumPlayed else 0.0,
        1.0 if st.energyAttached else 0.0,
        1.0 if st.retreated else 0.0,
        1.0 if st.stadium else 0.0,
        (sel.minCount or 0) / 5.0,
        (sel.maxCount or 0) / 5.0,
        min(len(sel.option), 30) / 30.0,
        (sel.remainDamageCounter or 0) / 10.0,
        (sel.remainEnergyCost or 0) / 5.0,
    ], dtype=np.float32)

    return np.concatenate([
        scalars,
        _onehot(sel.type, N_SELECT_TYPE),
        _onehot(sel.context, N_CONTEXT),
        _side_feats(mp),
        _side_feats(op),
        _mon_feats(my_act),
        _mon_feats(op_act),
        _weakness_pair(my_act, op_act),
    ])


STATE_DIM = 13 + N_SELECT_TYPE + N_CONTEXT + 2 * _SIDE_DIM + 2 * _MON_DIM + 2


def _target_feats(obs, opt) -> np.ndarray:
    """Stats of the in-play Pokemon an ATTACH/EVOLVE-style option points at."""
    st = obs.current
    me = st.yourIndex
    area, idx = opt.inPlayArea, opt.inPlayIndex
    if area is None or idx is None:
        return np.zeros(4, dtype=np.float32)
    p = st.players[me]
    mon = None
    if area == AreaType.ACTIVE and idx < len(p.active):
        mon = p.active[idx]
    elif area == AreaType.BENCH and idx < len(p.bench):
        mon = p.bench[idx]
    if mon is None:
        return np.zeros(4, dtype=np.float32)
    return np.array([
        1.0,
        (mon.hp or 0) / _MAX_HP,
        len(mon.energies or []) / 5.0,
        1.0 if area == AreaType.ACTIVE else 0.0,
    ], dtype=np.float32)


def _attack_feats(obs, opt) -> np.ndarray:
    """Damage/cost/KO features for an ATTACK option."""
    if opt.type != OptionType.ATTACK or opt.attackId is None:
        return np.zeros(5, dtype=np.float32)
    atk = ATTACK_DB.get(opt.attackId)
    if atk is None:
        return np.zeros(5, dtype=np.float32)
    st = obs.current
    me = st.yourIndex
    mp, op = st.players[me], st.players[1 - me]
    my_act = mp.active[0] if mp.active else None
    op_act = op.active[0] if op.active else None
    dmg = float(atk.damage)
    eff = dmg
    if my_act and op_act:
        mc, oc = CARD_DB.get(my_act.id), CARD_DB.get(op_act.id)
        if mc and oc and oc.weakness is not None and oc.weakness == mc.energyType:
            eff = dmg * 2
    opp_hp = (op_act.hp if op_act else 999)
    return np.array([
        1.0,
        dmg / _MAX_DMG,
        eff / _MAX_DMG,
        len(atk.energies) / 5.0,
        1.0 if eff >= opp_hp else 0.0,   # would KO
    ], dtype=np.float32)


def encode_options(obs: Observation) -> np.ndarray:
    st = obs.current
    me = st.yourIndex
    opts = obs.select.option
    rows = []
    for opt in opts:
        cid = resolve_card_id(obs, opt)
        mine = 1.0 if (opt.playerIndex is None or opt.playerIndex == me) else 0.0
        misc = np.array([
            mine,
            (opt.index or 0) / 20.0,
            (opt.number or 0) / 10.0,
            (opt.count or 0) / 5.0,
            (opt.toolIndex or 0) / 2.0,
            (opt.energyIndex or 0) / 5.0,
        ], dtype=np.float32)
        rows.append(np.concatenate([
            _onehot(opt.type, N_OPTION_TYPE),
            _onehot(opt.area, N_AREA),
            _card_attrs(cid),
            _target_feats(obs, opt),
            _attack_feats(obs, opt),
            misc,
        ]))
    if not rows:
        return np.zeros((0, OPTION_DIM), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


OPTION_DIM = N_OPTION_TYPE + N_AREA + _CARD_ATTR_DIM + 4 + 5 + 6
INPUT_DIM = STATE_DIM + OPTION_DIM


def encode(obs: Observation):
    """Return (state_vec[STATE_DIM], option_mat[n_options, OPTION_DIM])."""
    return encode_state(obs), encode_options(obs)


def build_inputs(obs: Observation) -> np.ndarray:
    """Per-option network input: state broadcast + option feats -> [n, INPUT_DIM]."""
    s, o = encode(obs)
    if o.shape[0] == 0:
        return np.zeros((0, INPUT_DIM), dtype=np.float32)
    return np.concatenate([np.repeat(s[None, :], o.shape[0], axis=0), o], axis=1)
