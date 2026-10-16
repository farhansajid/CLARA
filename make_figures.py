"""Build all figures (PDF) and LaTeX table bodies from ../results/*.csv."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(HERE, "..", "latex", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

ORDER = ["RR", "PRIO", "EDF", "MW", "EDF-PC", "CLARA"]
COL = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MRK = dict(zip(ORDER, ["o", "s", "^", "v", "D", "*"]))
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e2dc", "grid.linewidth": 0.5,
                     "legend.frameon": False, "pdf.fonttype": 42})
W1 = 3.5


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def sweep_plot(df, xcol, ycols, ylabels, xlabel, fname, logy=True, floor=1e-4):
    fig, axs = plt.subplots(1, len(ycols), figsize=(W1 * 2, 2.2))
    for ax, yc, yl in zip(axs, ycols, ylabels):
        for s in ORDER:
            d = df[df.scheme == s].groupby(xcol)[yc]
            m, e = d.mean(), d.apply(ci)
            y = np.maximum(m.values, floor) if logy else m.values
            ax.errorbar(m.index, y, yerr=e.values if not logy else None, color=COL[s], marker=MRK[s],
                        ms=4.5 if s != "CLARA" else 7, lw=2 if s == "CLARA" else 1.2, capsize=2,
                        label=s if s != "CLARA" else "CLARA (proposed)")
            if logy and s == "CLARA":
                ax.fill_between(m.index, np.maximum(m - e, floor), np.maximum(m + e, floor), color=COL[s], alpha=0.15, lw=0)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(bottom=floor * 0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(yl)
    axs[0].legend(ncol=3, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.32))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight")
    plt.close(fig)


def fmt(m, e, prec, pct=False):
    if pct:
        return f"{100 * m:.{prec}f}$\\pm${100 * e:.{prec}f}"
    return f"{m:.{prec}f}$\\pm${e:.{prec}f}"


def table_e1():
    df = pd.read_csv(os.path.join(RES, "e1_default.csv"))
    names = ["RR", "PRIO", "EDF", "MW", "EDF-PC", "CLARA", "CLARA-P", "CLARA-FP", "CLARA-O"]
    cols = [("miss1", 2, True), ("miss2", 2, True), ("miss3", 2, True), ("lat1_p99", 1, False),
            ("energy_uJ_per_pkt", 1, False), ("aoi_c2_ms", 1, False)]
    summary = []
    for tag in ("e1_walking", "e1_static"):
        lines = []
        for s in names:
            d = df[(df.tag == tag) & (df.scheme == s)]
            row = [s.replace("CLARA", "\\textbf{CLARA}") if s == "CLARA" else s]
            for c, p, pct in cols:
                row.append(fmt(d[c].mean(), ci(d[c]), p, pct))
                summary.append(dict(tag=tag, scheme=s, metric=c, mean=d[c].mean(), ci=ci(d[c])))
            lines.append(" & ".join(row) + " \\\\")
            if s == "CLARA":
                lines.append("\\midrule")
        with open(os.path.join(TAB, f"{tag}.tex"), "w") as f:
            f.write("\n".join(lines) + "\n")
    pd.DataFrame(summary).to_csv(os.path.join(RES, "e1_summary.csv"), index=False)


def table_ecg():
    df = pd.read_csv(os.path.join(RES, "ecg_app.csv"))
    lines, summ = [], []
    for s in ORDER:
        d = df[df.scheme == s]
        row = ["\\textbf{CLARA}" if s == "CLARA" else s,
               fmt(d.pkt_loss.mean(), ci(d.pkt_loss), 2, True),
               fmt(d.prd100.mean(), ci(d.prd100), 2), fmt(d.se100.mean() * 100, ci(d.se100) * 100, 2),
               fmt(d.f1_100.mean() * 100, ci(d.f1_100) * 100, 2),
               fmt(d.prd208.mean(), ci(d.prd208), 2), fmt(d.se208.mean() * 100, ci(d.se208) * 100, 2),
               fmt(d.energy_uJ_per_pkt.mean(), ci(d.energy_uJ_per_pkt), 1)]
        lines.append(" & ".join(row) + " \\\\")
    with open(os.path.join(TAB, "ecg.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    df.groupby("scheme").agg(["mean", ci]).to_csv(os.path.join(RES, "ecg_summary.csv"))


def fig_tradeoff():
    e4 = pd.read_csv(os.path.join(RES, "e4_tradeoff.csv"))
    e1 = pd.read_csv(os.path.join(RES, "e1_default.csv"))
    e1 = e1[e1.tag == "e1_walking"]
    fig, axs = plt.subplots(1, 2, figsize=(W1 * 2, 2.3))
    for ax, yc, yl in zip(axs, ["miss1", "miss_all"], ["C1 (critical) miss ratio", "Overall miss ratio"]):
        g = e4.groupby("V")
        ax.plot(g.energy_uJ_per_pkt.mean(), g[yc].mean(), "-", color=COL["CLARA"], marker="*", ms=7, lw=2,
                label="CLARA, varying $V$")
        for s in ORDER[:-1]:
            d = e1[e1.scheme == s]
            ax.plot(d.energy_uJ_per_pkt.mean(), d[yc].mean(), MRK[s], color=COL[s], ms=6, label=s)
        ax.set_xlabel("Energy per delivered packet ($\\mu$J)")
        ax.set_ylabel(yl)
    axs[0].legend(ncol=3, fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.32))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tradeoff.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_predictor():
    df = pd.read_csv(os.path.join(RES, "predictor_rmse.csv"))
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    lab = {"persistence": "Persistence (last RSSI)", "ridge": "Ridge (per age)", "mlp": "MLP (proposed)"}
    for k, (m, c) in enumerate(zip(["persistence", "ridge", "mlp"], ["#2a78d6", "#eb6834", "#008300"])):
        for sc, ls in (("walking", "-"), ("static", "--")):
            d = df[(df.model == m) & (df.scenario == sc)].groupby("age").rmse.mean()
            ax.plot(d.index * 5, d.values, ls, color=c, lw=1.5, marker=["o", "s", "*"][k], ms=3.5, markevery=3,
                    label=f"{lab[m]}, {sc}")
    ax.set_xlabel("Age of last channel observation (ms)")
    ax.set_ylabel("Prediction RMSE (dB)")
    ax.legend(fontsize=6.5, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "predictor.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_channel_example():
    import sys
    sys.path.insert(0, HERE)
    from wban.common import channel_traces
    from wban.sim import SimConfig
    L, obs = channel_traces(SimConfig(), 4242, n_nodes=10, sim_s=6.0)
    t = np.arange(L.shape[0]) * 5e-3
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    for i, name, c in ((0, "ECG (chest)", "#2a78d6"), (4, "Accelerometer (ankle)", "#eb6834")):
        ax.plot(t, L[:, i], color=c, lw=1.1, label=name)
        ax.plot(t[::20], obs[:, i], "o", color=c, ms=2.5, mfc="white", mew=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Large-scale loss (dB)")
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "channel_example.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    e2 = pd.read_csv(os.path.join(RES, "e2_nodes.csv"))
    sweep_plot(e2, "n_nodes", ["miss1", "miss2"], ["C1 (critical) miss ratio", "C2 (vital stream) miss ratio"],
               "Number of sensor nodes $N$", "nodes_miss.pdf")
    sweep_plot(e2, "n_nodes", ["energy_uJ_per_pkt", "aoi_c2_ms"], ["Energy per delivered packet ($\\mu$J)",
               "Average AoI of vital streams (ms)"], "Number of sensor nodes $N$", "nodes_energy_aoi.pdf", logy=False)
    e3 = pd.read_csv(os.path.join(RES, "e3_shadowing.csv"))
    sweep_plot(e3, "chan_pi_bad", ["miss1", "miss2"], ["C1 (critical) miss ratio", "C2 (vital stream) miss ratio"],
               "Body-shadowing probability $\\pi_B$", "shadow_miss.pdf")
    fig_tradeoff()
    fig_predictor()
    fig_channel_example()
    table_e1()
    table_ecg()
    print("figures and tables written")
