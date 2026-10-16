"""Run every experiment reported in the paper and write CSV files to ../results.

Usage: python run_experiments.py [--quick]
Evaluation seeds are 1..10 (ECG experiment: 1..5); predictor training seeds start at 1000,
predictor test seeds at 3000, hyper-parameter tuning used seeds 500..503 only.
"""
import os, sys, time, json, argparse
from dataclasses import replace, asdict
from multiprocessing import Pool
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from wban.sim import SimConfig, Scenario, run, metrics, SCHEMES
from wban.channel import ChannelParams
from wban.common import train_predictors, load_predictors, make_table, sig_idx_fn, channel_traces
from wban.predictor import build_dataset
from wban import ecg_app

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
PRED_PATH = os.path.join(RES, "predictors.pkl")
MAIN = ["RR", "PRIO", "EDF", "MW", "EDF-PC", "CLARA"]

_G = {}


def _init():
    _G["preds"] = load_predictors(PRED_PATH)
    _G["tab"] = make_table(SimConfig())


def make_cfg(over):
    over = dict(over)
    chan = replace(ChannelParams(), **over.pop("chan", {}))
    return replace(SimConfig(), chan=chan, **over)


def job(args):
    tag, over, seed, schemes, extra = args
    cfg = make_cfg(over)
    scn = Scenario(cfg, seed, _G["preds"])
    rows = []
    for s in schemes:
        Vs = extra.get("V_list", [None]) if s == "CLARA" else [None]
        for V in Vs:
            res = run(scn, s, _G["tab"], sig_idx_fn, V=V)
            m = metrics(res)
            m.update(tag=tag, seed=seed, scheme=s, V=cfg.V if V is None else V)
            for k, v in over.items():
                if k == "chan":
                    m.update({f"chan_{a}": b for a, b in v.items()})
                else:
                    m[k] = v
            if extra.get("ecg"):
                m["ecg_ok"] = ecg_delivery(res, 0)
            rows.append(m)
    return rows


def ecg_delivery(res, node):
    rec = res["rec"]
    m = (rec[:, 0] == node) & (rec[:, 5] >= 0)
    seq = rec[m, 5].astype(int)
    ok = rec[m, 4] > 0
    flags = np.zeros(seq.max() + 1, bool) if len(seq) else np.zeros(0, bool)
    flags[seq[ok]] = True
    return flags.tolist()


def run_pool(jobs, name):
    t0 = time.time()
    with Pool(os.cpu_count(), initializer=_init) as p:
        rows = [r for rr in p.imap_unordered(job, jobs) for r in rr]
    df = pd.DataFrame(rows)
    if "ecg_ok" in df:
        df.drop(columns="ecg_ok").to_csv(os.path.join(RES, f"{name}.csv"), index=False)
    else:
        df.to_csv(os.path.join(RES, f"{name}.csv"), index=False)
    print(f"{name}: {len(rows)} runs in {time.time() - t0:.0f}s", flush=True)
    return df


def exp_predictor(seeds):
    preds = load_predictors(PRED_PATH)
    rows = []
    for gait in (True, False):
        cfg = SimConfig(chan=replace(ChannelParams(), gait=gait))
        for s in seeds:
            L, obs = channel_traces(cfg, 3000 + s)
            X, y, a = build_dataset(L, obs, cfg.M)
            for kind, p in preds.items():
                err = y - p.predict_delta(X, a)
                for k in range(cfg.M):
                    rows.append(dict(scenario="walking" if gait else "static", seed=s, model=kind, age=k,
                                     rmse=float(np.sqrt(np.mean(err[a == k] ** 2)))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "predictor_rmse.csv"), index=False)
    print("predictor done", flush=True)


def exp_ecg(seeds, over):
    jobs = [("ecg", over, s, MAIN, {"ecg": True}) for s in seeds]
    with Pool(os.cpu_count(), initializer=_init) as p:
        rows = [r for rr in p.imap_unordered(job, jobs) for r in rr]
    x100, fs, beats = ecg_app.load_record100()
    x208, _ = ecg_app.load_record208_segment()
    b208_ref = ecg_app.detect_r(x208, fs)  # 208 has no bundled annotations: clean-signal detections as reference
    cfg = make_cfg(over)
    pkt_s = 1.0 / (4.5 * cfg.load)
    out = []
    for r in rows:
        ok = np.array(r["ecg_ok"], bool)
        n = min(len(ok) - 5, int(300 / pkt_s))
        ok = ok[:n]
        w0 = int((r["seed"] - 1) * 300 * fs)
        x = x100[w0:]
        xr, _ = ecg_app.receive(x, fs, ok, pkt_s)
        ref = beats[(beats >= w0) & (beats < w0 + len(xr))] - w0
        se, ppv, f1 = ecg_app.match(ref, ecg_app.detect_r(xr, fs), fs)
        xr8, _ = ecg_app.receive(x208, fs, ok, pkt_s)
        se8, ppv8, f18 = ecg_app.match(b208_ref[b208_ref < len(xr8)], ecg_app.detect_r(xr8, fs), fs)
        out.append(dict(seed=r["seed"], scheme=r["scheme"], pkt_loss=1 - ok.mean(), miss2=r["miss2"],
                        energy_uJ_per_pkt=r["energy_uJ_per_pkt"],
                        prd100=ecg_app.prd(x[:len(xr)], xr), se100=se, ppv100=ppv, f1_100=f1,
                        prd208=ecg_app.prd(x208[:len(xr8)], xr8), se208=se8, ppv208=ppv8, f1_208=f18))
    pd.DataFrame(out).to_csv(os.path.join(RES, "ecg_app.csv"), index=False)
    print("ecg done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    seeds = list(range(1, 4 if args.quick else 11))
    only = set(args.only.split(",")) if args.only else None
    want = lambda k: only is None or k in only
    with open(os.path.join(RES, "config_default.json"), "w") as f:
        json.dump(asdict(SimConfig()), f, indent=1, default=str)
    if want("train") or not os.path.exists(PRED_PATH):
        t0 = time.time()
        train_predictors(SimConfig(), PRED_PATH)
        print(f"predictors trained in {time.time() - t0:.0f}s", flush=True)
    if want("pred"):
        exp_predictor(seeds[:5])
    if want("e1"):
        jobs = [(f"e1_{g}", {"chan": {"gait": g == "walking"}}, s, SCHEMES, {})
                for g in ("walking", "static") for s in seeds]
        run_pool(jobs, "e1_default")
    if want("e2"):
        jobs = [("e2", {"n_nodes": n}, s, MAIN, {}) for n in (6, 9, 12, 15, 18, 21, 24) for s in seeds]
        run_pool(jobs, "e2_nodes")
    if want("e3"):
        jobs = [("e3", {"chan": {"pi_bad": pb}}, s, MAIN, {})
                for pb in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3) for s in seeds]
        run_pool(jobs, "e3_shadowing")
    if want("e4"):
        jobs = [("e4", {}, s, ["CLARA"], {"V_list": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]}) for s in seeds]
        run_pool(jobs, "e4_tradeoff")
    if want("e2ng"):
        jobs = [("e2ng", {"n_nodes": n}, s, ["CLARA-NG"], {}) for n in (6, 9, 12, 15, 18, 21, 24) for s in seeds]
        run_pool(jobs, "e2_noguard")
    if want("ecg"):
        exp_ecg(seeds[:5], {"n_nodes": 21, "sim_s": 300.0, "chan": {"pi_bad": 0.25}})
