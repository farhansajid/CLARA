"""On-body WBAN channel model used in all experiments.

Large-scale loss of node i at slot t (dB):
    L_i(t) = PL_i + L_impl + S_i(t) + B_i(t) + G_i(t)
  PL_i   : IEEE 802.15.6 CM3 path loss, a*log10(d_mm) + b (hospital room, 2.4 GHz)
  L_impl : fixed implementation/antenna-detuning margin near tissue
  S_i(t) : log-normal shadowing, first-order Gauss-Markov (AR(1)) process, std sigma_s
  B_i(t) : body-shadowing (NLOS) excess loss, two-state Gilbert-Elliott Markov chain
  G_i(t) : periodic gait-induced fluctuation for limb-mounted nodes (walking scenarios)
Small-scale fading: Nakagami-m power gain, i.i.d. per slot (unknown before transmission).
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ChannelParams:
    slot_s: float = 5e-3            # slot duration (s)
    cm3_a: float = 6.6              # CM3 path-loss slope (dB/decade of mm)
    cm3_b: float = 36.1             # CM3 intercept (dB)
    sigma_s: float = 3.8            # CM3 shadowing std (dB)
    tau_s: float = 0.30             # shadowing correlation time (s)
    impl_loss: float = 22.0         # antenna/body implementation margin (dB)
    pi_bad: float = 0.10            # stationary probability of body-shadow (NLOS) state
    bad_dur_s: float = 0.20         # mean sojourn in NLOS state (s)
    bad_loss_rng: tuple = (15.0, 30.0)  # per-node NLOS excess loss range (dB)
    gait: bool = True               # walking (True) or static subject
    gait_f: float = 0.9             # gait frequency (Hz)
    gait_amp: float = 5.0           # gait fluctuation amplitude for limb nodes (dB)
    nak_m: float = 2.0              # Nakagami-m small-scale fading parameter
    noise_dbm: float = -174 + 10 * np.log10(1e6) + 10.0  # kTB + NF over 1 MHz
    bw_hz: float = 1e6
    rate_bps: float = 242.9e3       # IEEE 802.15.6 NB PHY rate (2.4 GHz band)
    pkt_bits: int = 1000 + 7 * 8    # payload + MAC header bits


def gen_large_scale(dist_mm, limb, T, prm: ChannelParams, rng):
    """Return (T, N) array of large-scale loss in dB."""
    N = len(dist_mm)
    pl = prm.cm3_a * np.log10(dist_mm) + prm.cm3_b + prm.impl_loss
    # AR(1) shadowing
    rho = np.exp(-prm.slot_s / prm.tau_s)
    S = np.empty((T, N))
    S[0] = rng.normal(0, prm.sigma_s, N)
    w = rng.normal(0, prm.sigma_s * np.sqrt(1 - rho ** 2), (T, N))
    for t in range(1, T):
        S[t] = rho * S[t - 1] + w[t]
    # Gilbert-Elliott body shadowing
    B = np.zeros((T, N))
    if prm.pi_bad > 0:
        p_bg = prm.slot_s / prm.bad_dur_s
        p_gb = p_bg * prm.pi_bad / (1 - prm.pi_bad)
        excess = rng.uniform(*prm.bad_loss_rng, N)
        state = rng.random(N) < prm.pi_bad
        u = rng.random((T, N))
        for t in range(T):
            B[t] = state * excess
            flip = np.where(state, u[t] < p_bg, u[t] < p_gb)
            state = state ^ flip
    # gait component (limb nodes only), slight per-node phase and frequency jitter
    G = np.zeros((T, N))
    if prm.gait:
        tt = np.arange(T) * prm.slot_s
        f = prm.gait_f * (1 + 0.03 * rng.standard_normal(N))
        ph = rng.uniform(0, 2 * np.pi, N)
        G = (prm.gait_amp * limb)[None, :] * np.sin(2 * np.pi * f[None, :] * tt[:, None] + ph[None, :])
    return pl[None, :] + S + B + G


def per_from_snr_db(snr_db, prm: ChannelParams):
    """Packet error rate for pi/2-DBPSK (non-coherent DBPSK BER = 0.5 exp(-Eb/N0))."""
    ebn0 = 10 ** (np.asarray(snr_db) / 10) * prm.bw_hz / prm.rate_bps
    ber = 0.5 * np.exp(-ebn0)
    return 1 - (1 - ber) ** prm.pkt_bits


def fading_db(rng, m, size):
    return 10 * np.log10(rng.gamma(m, 1.0 / m, size))


class SuccessTable:
    """P(success | predicted margin, residual std) under Nakagami-m fading + Gaussian
    prediction error, tabulated by Monte Carlo; margin = P_tx - L_hat - noise (dB)."""

    def __init__(self, prm: ChannelParams, sigmas, n_mc=20000, seed=7):
        rng = np.random.default_rng(seed)
        self.grid = np.arange(-30, 60.01, 0.25)
        self.sigmas = np.asarray(sigmas, float)
        fad = fading_db(rng, prm.nak_m, n_mc)
        z = rng.standard_normal(n_mc)
        tab = np.empty((len(self.sigmas), len(self.grid)))
        for k, s in enumerate(self.sigmas):
            snr = self.grid[:, None] + fad[None, :] - s * z[None, :]
            tab[k] = (1 - per_from_snr_db(snr, prm)).mean(1)
        self.tab = tab

    def __call__(self, margin, sigma_idx):
        idx = np.clip(np.round((margin - self.grid[0]) / 0.25).astype(int), 0, len(self.grid) - 1)
        return self.tab[sigma_idx, idx]
