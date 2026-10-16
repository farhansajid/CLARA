"""Slot-level simulator of a hub-coordinated (polled / scheduled-access) WBAN uplink.

One uplink packet per slot (5 ms incl. MAC header, ACK and guard time). Packets are
retransmitted until delivered or until their deadline expires (then counted as missed).
"""
import heapq
import numpy as np
from dataclasses import dataclass, field
from .channel import ChannelParams, gen_large_scale, per_from_snr_db, fading_db, SuccessTable

C1, C2, C3 = 0, 1, 2  # critical (alarms), high (vital streams), normal (context/telemetry)

# name, class, rate (pkt/s), deadline (ms), alarm-capable, limb-mounted, distance range (mm)
NODE_TEMPLATES = [
    ("ECG",     C2, 4.5,  250, True,  0, (150, 300)),
    ("EEG",     C2, 25.0, 120, True,  0, (400, 600)),
    ("SpO2",    C2, 2.0,  500, True,  1, (450, 650)),
    ("EMG",     C3, 12.0, 200, False, 1, (400, 600)),
    ("Acc-ank", C3, 5.0,  400, True,  1, (900, 1100)),
    ("Temp",    C3, 0.5, 2000, False, 0, (100, 300)),
    ("CGM",     C3, 0.2, 2000, True,  0, (150, 300)),
    ("BP",      C3, 1.0, 1000, False, 1, (450, 650)),
    ("Resp",    C3, 4.0,  500, False, 0, (100, 250)),
    ("Acc-wr",  C3, 5.0,  400, False, 1, (450, 650)),
]
ALARM_DEADLINE_MS = 50
ALARM_BURST = 3

POWERS_DBM = np.array([-20.0, -10.0, -5.0, 0.0, 4.0])
P_REF_IDX = 3  # 0 dBm, used by fixed-power schemes


@dataclass
class SimConfig:
    n_nodes: int = 12
    sim_s: float = 120.0
    load: float = 1.3
    alarm_rate: float = 0.2      # alarm episodes per second per alarm-capable node
    M: int = 20                  # superframe length in slots (channel observation period)
    sigma_obs: float = 1.0       # RSSI observation noise (dB)
    chan: ChannelParams = field(default_factory=ChannelParams)
    # radio energy model (BLE-class SoC, 3 V supply)
    vdd: float = 3.0
    i0_ma: float = 3.5
    k_ma_per_mw: float = 1.8
    i_rx_ma: float = 5.4
    t_ack_s: float = 0.5e-3
    # proposed scheme
    V: float = 0.5
    omega: tuple = (50.0, 5.0, 1.0)
    eps: tuple = (1e-3, 1e-2, 5e-2)
    kappa: float = 3.0
    z_max: float = 3.0           # cap on virtual queues (avoids priority inversion when infeasible)
    guard: bool = True           # criticality guard: C1 packets take precedence when present


def tx_energy_j(cfg: SimConfig):
    t_pkt = cfg.chan.pkt_bits / cfg.chan.rate_bps
    i_tx = cfg.i0_ma + cfg.k_ma_per_mw * 10 ** (POWERS_DBM / 10)
    return cfg.vdd * i_tx * 1e-3 * t_pkt + cfg.vdd * cfg.i_rx_ma * 1e-3 * cfg.t_ack_s


def build_topology(n, rng):
    nodes = []
    for i in range(n):
        name, cls, rate, dl, alarm, limb, (d0, d1) = NODE_TEMPLATES[i % len(NODE_TEMPLATES)]
        nodes.append(dict(name=name if i < len(NODE_TEMPLATES) else f"{name}{i // len(NODE_TEMPLATES) + 1}",
                          cls=cls, rate=rate, dl_ms=dl, alarm=alarm, limb=limb, dist=rng.uniform(d0, d1)))
    return nodes


def gen_traffic(nodes, cfg: SimConfig, T, rng):
    """List of arrivals per slot: (node, cls, deadline_slots, app_seq). app_seq=-1 for alarms."""
    slot = cfg.chan.slot_s
    arr = [[] for _ in range(T)]
    for i, nd in enumerate(nodes):
        period = 1.0 / (nd["rate"] * cfg.load)
        t0 = rng.uniform(0, period)
        times = np.arange(t0, T * slot, period)
        times = times + rng.normal(0, 0.02 * period, len(times))  # sampling-clock jitter
        dls = max(1, int(round(nd["dl_ms"] * 1e-3 / slot)))
        for s, tt in enumerate(times):
            k = int(tt / slot)
            if 0 <= k < T:
                arr[k].append((i, nd["cls"], dls, s))
        if nd["alarm"]:
            n_ev = rng.poisson(cfg.alarm_rate * T * slot)
            dla = int(round(ALARM_DEADLINE_MS * 1e-3 / slot))
            for k in rng.integers(0, T, n_ev):
                for _ in range(ALARM_BURST):
                    arr[k].append((i, C1, dla, -1))
    return arr


class Scenario:
    """Everything random that is shared by all schedulers for one seed (common random numbers)."""

    def __init__(self, cfg: SimConfig, seed, predictors):
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.T = int(round(cfg.sim_s / cfg.chan.slot_s))
        self.nodes = build_topology(cfg.n_nodes, rng)
        dist = np.array([n["dist"] for n in self.nodes])
        limb = np.array([n["limb"] for n in self.nodes], float)
        self.L = gen_large_scale(dist, limb, self.T, cfg.chan, rng)
        from .predictor import observe
        self.obs = observe(self.L, cfg.M, cfg.sigma_obs, rng)
        self.arrivals = gen_traffic(self.nodes, cfg, self.T, rng)
        # per (slot,node) fading realisation and success uniform, identical for all schemes
        self.fad = fading_db(rng, cfg.chan.nak_m, (self.T, cfg.n_nodes))
        self.u = rng.random((self.T, cfg.n_nodes))
        self.est = {}
        for name, pred in predictors.items():
            self.est[name] = pred.predict_trace(self.obs, self.T)
        self.est["oracle"] = (self.L, np.zeros(self.T))


SCHEMES = ["RR", "PRIO", "EDF", "MW", "EDF-PC", "CLARA", "CLARA-P", "CLARA-FP", "CLARA-O"]


def run(scn: Scenario, scheme, table: SuccessTable, sig_idx_fn, V=None):
    cfg = scn.cfg
    V = cfg.V if V is None else V
    T, N = scn.T, cfg.n_nodes
    noise = cfg.chan.noise_dbm
    Etx = tx_energy_j(cfg)
    Eref = Etx[P_REF_IDX]
    omega = np.array(cfg.omega)
    eps = np.array(cfg.eps)
    dl_class = {}
    est_name = {"MW": "persistence", "EDF-PC": "persistence", "CLARA": "mlp",
                "CLARA-P": "persistence", "CLARA-FP": "mlp", "CLARA-O": "oracle", "CLARA-NG": "mlp"}.get(scheme)
    if est_name is not None:
        Lh, sh = scn.est[est_name]
        sidx = sig_idx_fn(sh)
    queues = [[] for _ in range(N)]  # heap of (cls, deadline, gen, pid, app_seq)
    pid = 0
    rec = []  # (node, cls, gen, finish, delivered, app_seq)
    energy = np.zeros(N)
    attempts = np.zeros(N, int)
    Z = np.zeros(3)
    rr_ptr = 0
    pw_idx = np.arange(len(POWERS_DBM))
    for t in range(T):
        for (i, c, dls, aseq) in scn.arrivals[t]:
            heapq.heappush(queues[i], (c, t + dls, t, pid, aseq))
            dl_class[pid] = dls
            pid += 1
        # drop expired packets (deadline passed)
        for i in range(N):
            q = queues[i]
            if q and min(p[1] for p in q) <= t:
                keep = []
                for p in q:
                    if p[1] <= t:
                        rec.append((i, p[0], p[2], t, False, p[4]))
                        Z[p[0]] = max(Z[p[0]] + 1 - eps[p[0]], 0)
                    else:
                        keep.append(p)
                heapq.heapify(keep)
                queues[i] = keep
        busy = [i for i in range(N) if queues[i]]
        if not busy:
            continue
        sel, pidx = None, P_REF_IDX
        if scheme == "RR":
            for k in range(N):
                j = (rr_ptr + k) % N
                if queues[j]:
                    sel = j
                    rr_ptr = (j + 1) % N
                    break
        elif scheme == "PRIO":
            sel = min(busy, key=lambda i: (queues[i][0][0], queues[i][0][2]))
        elif scheme == "EDF":
            sel = min(busy, key=lambda i: queues[i][0][1])
        elif scheme == "MW":
            m = POWERS_DBM[P_REF_IDX] - Lh[t, busy] - noise
            ps = table(m, sidx[t])
            w = np.array([len(queues[i]) for i in busy]) * omega[[queues[i][0][0] for i in busy]]
            sel = busy[int(np.argmax(w * ps))]
        elif scheme == "EDF-PC":
            sel = min(busy, key=lambda i: queues[i][0][1])
            m = POWERS_DBM - Lh[t, sel] - noise
            ok = np.nonzero(table(m, sidx[t]) >= 0.99)[0]
            pidx = int(ok[0]) if len(ok) else len(POWERS_DBM) - 1
        else:  # CLARA family: drift-plus-penalty with deadline-urgency weights
            if cfg.guard and scheme != "CLARA-NG":
                crit = [i for i in busy if queues[i][0][0] == C1]
                if crit:
                    busy = crit
            b = np.array(busy)
            hol = [queues[i][0] for i in busy]
            cls = np.array([h[0] for h in hol])
            slack = np.array([h[1] - t for h in hol], float)
            D = np.array([dl_class[h[3]] for h in hol], float)
            W = (1 + (Z[cls] if scheme == "CLARA-NG" else np.minimum(Z[cls], cfg.z_max))) * omega[cls] * np.exp(cfg.kappa * (1 - slack / D))
            pset = pw_idx if scheme != "CLARA-FP" else np.array([P_REF_IDX])
            m = POWERS_DBM[pset][None, :] - Lh[t, b][:, None] - noise
            ps = table(m, sidx[t])
            score = W[:, None] * ps - V * (Etx[pset] / Eref)[None, :]
            k = np.unravel_index(int(np.argmax(score)), score.shape)
            if score[k] <= 0:
                continue  # defer: expected utility below energy cost (predicted deep fade)
            sel, pidx = busy[k[0]], int(pset[k[1]])
        # transmit HOL packet of node sel
        attempts[sel] += 1
        energy[sel] += Etx[pidx]
        snr = POWERS_DBM[pidx] - scn.L[t, sel] - noise + scn.fad[t, sel]
        if scn.u[t, sel] < 1 - per_from_snr_db(snr, cfg.chan):
            c, dl, g, p_, aseq = heapq.heappop(queues[sel])
            rec.append((sel, c, g, t + 1, True, aseq))
            Z[c] = max(Z[c] - eps[c], 0)
    # packets still queued at end: censored, not counted
    return dict(rec=np.array(rec, dtype=float).reshape(-1, 6), energy=energy, attempts=attempts,
                T=T, slot=cfg.chan.slot_s, nodes=scn.nodes)


def metrics(res):
    rec, slot = res["rec"], res["slot"]
    out = {}
    node, cls, gen, fin, ok = rec[:, 0].astype(int), rec[:, 1].astype(int), rec[:, 2], rec[:, 3], rec[:, 4] > 0
    for c in range(3):
        m = cls == c
        out[f"miss{c + 1}"] = 1 - ok[m].mean() if m.any() else np.nan
        out[f"n{c + 1}"] = int(m.sum())
        lat = (fin[m & ok] - gen[m & ok]) * slot * 1e3
        out[f"lat{c + 1}_mean"] = lat.mean() if len(lat) else np.nan
        out[f"lat{c + 1}_p99"] = np.percentile(lat, 99) if len(lat) else np.nan
    out["miss_all"] = 1 - ok.mean()
    nd = ok.sum()
    out["energy_uJ_per_pkt"] = res["energy"].sum() / max(nd, 1) * 1e6
    out["energy_mJ_total"] = res["energy"].sum() * 1e3
    out["attempts_per_pkt"] = res["attempts"].sum() / max(nd, 1)
    N = len(res["nodes"])
    dr = np.array([ok[node == i].mean() if (node == i).any() else 1.0 for i in range(N)])
    out["jain"] = dr.sum() ** 2 / (N * (dr ** 2).sum())
    # average Age of Information of the periodic vital-sign streams (C2), ms
    T = res["T"]
    aois = []
    for i, nd_ in enumerate(res["nodes"]):
        if nd_["cls"] != C2:
            continue
        m = (node == i) & ok & (rec[:, 5] >= 0)
        latest = np.full(T + 1, np.nan)
        if m.any():
            f, g = fin[m].astype(int), gen[m]
            upd = np.full(T + 1, -np.inf)
            np.maximum.at(upd, np.minimum(f, T), g)
            latest = np.maximum.accumulate(upd)
        age = np.arange(T + 1) - latest
        valid = np.isfinite(age)
        aois.append(age[valid][T // 10:].mean() * slot * 1e3 if valid.any() else np.nan)
    out["aoi_c2_ms"] = float(np.nanmean(aois)) if aois else np.nan
    return out
