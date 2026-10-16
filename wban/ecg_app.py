"""Application-level impact of packet losses on real ECG (MIT-BIH Arrhythmia Database).

Each ECG-node packet carries one packetisation interval of samples. A packet missed or
delivered after its deadline is unusable for real-time monitoring; the hub conceals the gap
by linear interpolation. We report PRD (signal fidelity) and R-peak detection sensitivity /
PPV / F1 against the cardiologist beat annotations (record 100) using a Pan-Tompkins-style
detector applied identically to the clean and the received signal.
"""
import os
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")


def load_record100():
    import wfdb
    r = wfdb.rdrecord(os.path.join(DATA, "100"))
    a = wfdb.rdann(os.path.join(DATA, "100"), "atr")
    beats = np.array([s for s, sym in zip(a.sample, a.symbol) if sym in BEAT_SYMBOLS])
    return r.p_signal[:, 0], r.fs, beats


def load_record208_segment():
    """5-min excerpt of MIT-BIH record 208 (MLII) as distributed with scipy.datasets."""
    with np.load(os.path.join(DATA, "scipy_ecg_mitdb208.dat")) as f:
        ecg = f["ecg"].astype(int)
    return (ecg - 1024) / 200.0, 360  # same conversion as scipy.datasets.electrocardiogram


def detect_r(x, fs):
    b, a = butter(2, [5 / (fs / 2), 15 / (fs / 2)], "band")
    y = filtfilt(b, a, x)
    d = np.gradient(y) ** 2
    w = int(0.15 * fs)
    mwi = np.convolve(d, np.ones(w) / w, "same")
    thr = 0.3 * np.percentile(mwi, 99)
    pk, _ = find_peaks(mwi, height=thr, distance=int(0.25 * fs))
    # refine to the local max of the band-passed signal
    out = []
    for p in pk:
        lo, hi = max(p - w, 0), min(p + w, len(x))
        out.append(lo + int(np.argmax(np.abs(y[lo:hi]))))
    return np.unique(np.array(out, int))


def match(ref, det, fs, tol_s=0.15):
    tol = int(tol_s * fs)
    used = np.zeros(len(det), bool)
    tp = 0
    j = 0
    for r in ref:
        while j < len(det) and det[j] < r - tol:
            j += 1
        k = j
        while k < len(det) and det[k] <= r + tol:
            if not used[k]:
                used[k] = True
                tp += 1
                break
            k += 1
    fn = len(ref) - tp
    fp = len(det) - used.sum()
    se = tp / max(tp + fn, 1)
    ppv = tp / max(tp + fp, 1)
    return se, ppv, 2 * se * ppv / max(se + ppv, 1e-12)


def receive(x, fs, pkt_ok, pkt_s):
    """Rebuild signal from per-packet delivery flags; gaps concealed by linear interpolation."""
    n = int(round(pkt_s * fs * len(pkt_ok)))
    x = x[:n].copy()
    mask = np.ones(n, bool)
    edges = np.round(np.arange(len(pkt_ok) + 1) * pkt_s * fs).astype(int)
    for k, ok in enumerate(pkt_ok):
        if not ok:
            mask[edges[k]:edges[k + 1]] = False
    if (~mask).any() and mask.any():
        idx = np.arange(n)
        x[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
    return x, mask


def prd(x, xr):
    x0 = x - x.mean()
    return 100 * np.sqrt(np.sum((x - xr) ** 2) / np.sum(x0 ** 2))
