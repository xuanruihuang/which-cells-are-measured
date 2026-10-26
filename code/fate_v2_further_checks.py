"""Two of the further checks in Supplementary Table S2 (fate v2, 2026-10-26), from saved outputs only:
(1) within the most essential stratum (and the others), Spearman correlation of each perturbation's escape fraction with
    its number of cells (raw counts), per cell type, with a 95% bootstrap interval over genes (1,000 draws);
(3) hESC against K562, change in agreement of direction when hESC is restricted to knocked-down cells, without and with
    the classification control (thinning null) subtracted: median per-gene change with a 95% bootstrap interval
    (2,000 draws), by stratum.
Inputs: metrics_mixture.json (fate_mixture.py: hESC, K562 day 8), metrics_mixture_more.json (fate_mixture_more.py:
K562 day 6, RPE1, Jurkat, HepG2), c34_rows.npz (fate_c34_agree.py). The random streams and the order of draws are those
of the run that produced the table, so the intervals are reproduced exactly.
Usage: fate_v2_further_checks.py <metrics_mixture.json> <metrics_mixture_more.json> <c34_rows.npz> <out json>."""
import sys, json
import numpy as np
from scipy.stats import spearmanr

M1, M2, C34, OUT = sys.argv[1:5]
M = json.load(open(M1)); MM = json.load(open(M2))
R = {"hESC": M["rows"]["hESC"], "K562_d8": M["rows"]["K562_gw"], "K562_d6": MM["rows"]["K562ess"], "RPE1": MM["rows"]["RPE1"],
     "Jurkat": MM["rows"]["Jurkat"], "HepG2": MM["rows"]["HepG2"]}
T = [("<-1", -99, -1), ("-1..-0.5", -1, -0.5), ("-0.5..-0.2", -0.5, -0.2), (">-0.2", -0.2, 99)]
res = {"rules": __doc__, "e_vs_cells": {}, "direction": {}}
rng = np.random.default_rng(0)
for c, rows in R.items():
    res["e_vs_cells"][c] = {}
    for lab, lo, hi in T:
        s = [r for r in rows if r["essentiality"] is not None and np.isfinite(r["essentiality"]) and lo <= r["essentiality"] < hi]
        if len(s) < 15:
            res["e_vs_cells"][c][lab] = dict(n=len(s)); continue
        e = np.array([r["nb"]["e"] for r in s]); n = np.array([r["n"] for r in s], float)
        rho = spearmanr(n, e).correlation
        bs = [spearmanr(n[i], e[i]).correlation for i in (rng.integers(0, len(s), len(s)) for _ in range(1000))]
        res["e_vs_cells"][c][lab] = dict(n=len(s), rho=float(rho), ci95=[float(x) for x in np.percentile(bs, [2.5, 97.5])])
    print(c, json.dumps(res["e_vs_cells"][c]), flush=True)
z = np.load(C34); E = z["essentiality"]; A = z["rho_all"]; K = z["rho_kd"]; N = z["rho_kd_null2"]
rng = np.random.default_rng(1)


def ci(x):
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(2000)]
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


for lab, lo, hi in (("<-1", -9, -1), ("-1..-0.5", -1, -0.5), ("-0.5..-0.2", -0.5, -0.2), (">-0.2", -0.2, 9), ("all", -9, 9)):
    m = np.isfinite(E) & (E >= lo) & (E < hi) & np.isfinite(A) & np.isfinite(K); mn = m & np.isfinite(N)
    d1 = K[m] - A[m]; rec = dict(n=int(m.sum()), change=float(np.median(d1)), ci95=ci(d1))
    if mn.sum() > 4:
        d2 = N[mn] - A[mn]; rec.update(n_control=int(mn.sum()), change_control_subtracted=float(np.median(d2)), ci95_control_subtracted=ci(d2))
    res["direction"][lab] = rec
    print("direction", lab, json.dumps(rec), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
