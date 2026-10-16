"""Paired (common-random-number) comparisons of CLARA against each baseline; writes ../results/paired_tests.csv."""
import os
import pandas as pd
from scipy import stats

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
d = pd.read_csv(os.path.join(RES, "e1_default.csv"))
rows = []
for tag in ("e1_walking", "e1_static"):
    x = d[d.tag == tag].pivot(index="seed", columns="scheme")
    for b in ("RR", "PRIO", "EDF", "MW", "EDF-PC"):
        for v in ("miss1", "miss2", "miss3", "energy_uJ_per_pkt", "aoi_c2_ms"):
            a, c = x[v]["CLARA"], x[v][b]
            p = stats.wilcoxon(a, c).pvalue if (a != c).any() else 1.0
            rows.append(dict(scenario=tag, baseline=b, metric=v, clara=a.mean(), base=c.mean(),
                             rel_change_pct=100 * (a.mean() / c.mean() - 1), clara_better_seeds=int((a < c).sum()),
                             n_seeds=len(a), wilcoxon_p=p))
e2 = pd.read_csv(os.path.join(RES, "e2_nodes.csv"))
e2 = e2[e2.n_nodes <= 21]
for b in ("MW", "EDF-PC"):
    for v in ("miss1", "energy_uJ_per_pkt"):
        x = e2.pivot_table(index=["n_nodes", "seed"], columns="scheme", values=v)
        rows.append(dict(scenario="e2_N6-21_pooled", baseline=b, metric=v, clara=x["CLARA"].mean(), base=x[b].mean(),
                         rel_change_pct=100 * (x["CLARA"].mean() / x[b].mean() - 1),
                         clara_better_seeds=int((x["CLARA"] < x[b]).sum()), n_seeds=len(x),
                         wilcoxon_p=stats.wilcoxon(x["CLARA"], x[b]).pvalue))
pd.DataFrame(rows).to_csv(os.path.join(RES, "paired_tests.csv"), index=False)
print(pd.DataFrame(rows).tail(4).to_string())
