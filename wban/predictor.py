"""Lightweight learned large-scale channel predictor run at the hub.

The hub observes each node's large-scale loss once per superframe (every M slots,
from beacon/poll responses; RSSI averaged over fast fading, observation noise sigma_obs).
Between observations, the scheduler needs the loss at the current slot, age a = t mod M
slots after the last observation. Three estimators are compared:
  * persistence  : L_hat(t) = last observation (what CSI-based schemes use today)
  * ridge        : per-age linear predictor on the last K observations
  * mlp          : one small MLP shared across ages (age is an input feature)
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

K_HIST = 10


def observe(L, M, sigma_obs, rng):
    obs_idx = np.arange(0, L.shape[0], M)
    return L[obs_idx] + rng.normal(0, sigma_obs, (len(obs_idx), L.shape[1]))


def _features(obs, j, ref):
    """Features for observation index array j (per node column), ref = per-node level."""
    feats = [obs[j] - ref]
    for k in range(1, K_HIST):
        jj = np.maximum(j - k, 0)
        feats.append(obs[j] - obs[jj])
    return np.stack(feats, -1)


def build_dataset(L, obs, M):
    T, N = L.shape
    t = np.arange(K_HIST * M, T)
    j = t // M
    a = t - j * M
    ref = np.cumsum(obs, 0) / np.arange(1, len(obs) + 1)[:, None]  # causal running mean
    X, y, ages = [], [], []
    for i in range(N):
        f = _features(obs[:, i], j, ref[j, i])
        X.append(f)
        y.append(L[t, i] - obs[j, i])
        ages.append(a)
    return np.concatenate(X), np.concatenate(y), np.concatenate(ages)


class ChannelPredictor:
    def __init__(self, kind, M):
        self.kind, self.M = kind, M
        self.models = None
        self.res_std = np.zeros(M)

    def fit(self, X, y, a):
        if self.kind == "ridge":
            self.models = [Ridge(alpha=1.0).fit(X[a == k], y[a == k]) for k in range(self.M)]
        elif self.kind == "mlp":
            Xa = np.column_stack([X, a / self.M])
            self.models = make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(32, 16), max_iter=60, early_stopping=True, random_state=0)).fit(Xa, y)
        return self

    def predict_delta(self, X, a):
        if self.kind == "persistence":
            return np.zeros(len(X))
        if self.kind == "ridge":
            out = np.empty(len(X))
            for k in range(self.M):
                m = a == k
                if m.any():
                    out[m] = self.models[k].predict(X[m])
            return out
        return self.models.predict(np.column_stack([X, a / self.M]))

    def calibrate(self, X, y, a):
        r = y - self.predict_delta(X, a)
        self.res_std = np.array([r[a == k].std() for k in range(self.M)])
        return self

    def predict_trace(self, obs, T):
        """Return (L_hat (T,N), sigma_hat (T,)) for a full trace, causally."""
        N = obs.shape[1]
        t = np.arange(T)
        j = t // self.M
        a = t - j * self.M
        ref = np.cumsum(obs, 0) / np.arange(1, len(obs) + 1)[:, None]
        Lh = np.empty((T, N))
        for i in range(N):
            X = _features(obs[:, i], j, ref[j, i])
            Lh[:, i] = obs[j, i] + self.predict_delta(X, a)
        return Lh, self.res_std[a]
