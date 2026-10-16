import pickle
import numpy as np
from dataclasses import replace
from .channel import ChannelParams, gen_large_scale, SuccessTable
from .predictor import ChannelPredictor, observe, build_dataset
from .sim import SimConfig, build_topology

SIGMA_GRID = np.arange(0, 12.01, 0.25)


def sig_idx_fn(sh):
    return np.clip(np.round(np.asarray(sh) / 0.25).astype(int), 0, len(SIGMA_GRID) - 1)


def make_table(cfg: SimConfig):
    return SuccessTable(cfg.chan, SIGMA_GRID)


def channel_traces(cfg: SimConfig, seed, n_nodes=10, sim_s=120.0):
    rng = np.random.default_rng(seed)
    nodes = build_topology(n_nodes, rng)
    T = int(round(sim_s / cfg.chan.slot_s))
    L = gen_large_scale(np.array([n["dist"] for n in nodes]), np.array([n["limb"] for n in nodes], float),
                        T, cfg.chan, rng)
    return L, observe(L, cfg.M, cfg.sigma_obs, rng)


def training_conditions():
    """Mixed conditions used for training; seeds disjoint from evaluation seeds (>=1000)."""
    conds = []
    for gait in (True, False):
        for pb in (0.05, 0.1, 0.2):
            conds.append(replace(ChannelParams(), gait=gait, pi_bad=pb))
    return conds


def train_predictors(base_cfg: SimConfig, out_path=None, n_train=4, n_val=2):
    Xs, ys, As, Xv, yv, Av = [], [], [], [], [], []
    seed = 1000
    for chan in training_conditions():
        cfg = replace(base_cfg, chan=chan)
        for r in range(n_train + n_val):
            L, obs = channel_traces(cfg, seed)
            seed += 1
            X, y, a = build_dataset(L, obs, cfg.M)
            (Xs if r < n_train else Xv).append(X)
            (ys if r < n_train else yv).append(y)
            (As if r < n_train else Av).append(a)
    X, y, a = map(np.concatenate, (Xs, ys, As))
    sub = np.random.default_rng(0).choice(len(X), min(len(X), 300000), replace=False)
    X, y, a = X[sub], y[sub], a[sub]
    Xv_, yv_, av_ = map(np.concatenate, (Xv, yv, Av))
    preds = {}
    for kind in ("persistence", "ridge", "mlp"):
        p = ChannelPredictor(kind, base_cfg.M).fit(X, y, a).calibrate(Xv_, yv_, av_)
        preds[kind] = p
    if out_path:
        with open(out_path, "wb") as f:
            pickle.dump(preds, f)
    return preds


def load_predictors(path):
    with open(path, "rb") as f:
        return pickle.load(f)
