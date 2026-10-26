"""Fate v2 gap plan item 5 (readings stated before running). A100.
After restricting both sides to knocked-down cells, hESC responds at a fifth to a third of K562 on the genes whose
knockdown depletes hESCs. Is the knockdown of the remaining hESC knocked-down cells shallower than in K562?
Targets and magnitudes: the V1 rows of fate_rev_samescale.py (h_kd = hESC knocked-down cells minus the thinning null,
guide parities; k_kd = K562 knocked-down cells minus each screen's thinning null, day-6 x day-8 cross-signal).
Knocked-down depth per target = own-target counts / expected control-level counts of the knocked-down group:
hESC from pan_agg_kdgroup counts (group 0), K562 from both budget extractions (group 0, all parities, halves, phases).
Reported: median depth on each side; ratio of medians h_kd / k_kd overall, by hESC abundance bin, and on targets whose
depths differ by at most 0.05; Spearman of log2(h_kd / k_kd) against hESC depth minus K562 depth (targets with both
magnitudes positive).
Usage: fate_v2_kd_depth.py <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr

import config
FI = config.get("FATE_ROOT"); KDG = config.get("PAN_KDGROUP")


def rom(a, b, rng, n=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = np.median(a) / np.median(b)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a)); mb = np.median(b[i])
        if mb > 0:
            bs.append(np.median(a[i]) / mb)
    return dict(n=int(len(a)), ratio_of_medians=float(r), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); rng = np.random.default_rng(11)
    V1 = json.load(open(FI + "/rev_samescale/metrics_rev_samescale.json"))["V1_rows"]
    idx = json.load(open(KDG + "/index.json")); pt = idx["targets"]; T = len(pt)
    C = np.load(KDG + "/counts.npy").reshape(T, 2, 3, 4)
    def cnt(d, lab):   # counts only (the raw sums of these extractions are no longer kept); layout as fate_budget.Ctx
        tl = json.load(open("%s/%s_index.json" % (d, lab)))["targets"]
        return np.load("%s/%s_cnt.npy" % (d, lab)).reshape(len(tl), 2, 2, 3, 3, 4), {t: k for k, t in enumerate(tl)}
    cA, tA = cnt(FI + "/budget", "K562"); cB, tB = cnt(FI + "/budget_d8", "K562d8")
    rows = []
    for r in V1:
        t = r["target"]
        if r["h_kd"] is None or r["k_kd"] is None or not (np.isfinite(r["h_kd"]) and np.isfinite(r["k_kd"])):
            continue
        ti = pt.index(t)
        dh = float(C[ti, :, 0, 2].sum() / max(C[ti, :, 0, 3].sum(), 1e-12))
        ck = cA[tA[t]][..., 0, :].reshape(-1, 4).sum(0) + cB[tB[t]][..., 0, :].reshape(-1, 4).sum(0)
        dk = float(ck[2] / max(ck[3], 1e-12))
        rows.append(dict(target=t, essentiality=r["essentiality"], norm=r["norm"], h_kd=r["h_kd"], k_kd=r["k_kd"], depth_h=dh, depth_k=dk,
                         n_kd_h=r["n_kd_h"], n_kd_k=int(ck[0])))
    D = {k: np.array([np.nan if x[k] is None else x[k] for x in rows], float) for k in ("norm", "h_kd", "k_kd", "depth_h", "depth_k")}
    res = {"rules": __doc__, "n": len(rows)}
    res["depth_median_hESC"] = float(np.median(D["depth_h"])); res["depth_median_K562"] = float(np.median(D["depth_k"]))
    res["depth_paired_median_difference"] = float(np.median(D["depth_h"] - D["depth_k"]))
    res["ratio_all"] = rom(D["h_kd"], D["k_kd"], rng)
    for nm, lo, hi in (("le_0.05", -1, 0.05), ("0.05_0.2", 0.05, 0.2), ("0.2_0.5", 0.2, 0.5), ("gt_0.5", 0.5, 99)):
        m = (D["norm"] > lo) & (D["norm"] <= hi)
        res["bin_" + nm] = dict(n=int(m.sum()), depth_h=float(np.median(D["depth_h"][m])) if m.any() else None,
                                depth_k=float(np.median(D["depth_k"][m])) if m.any() else None,
                                ratio=rom(D["h_kd"][m], D["k_kd"][m], rng) if m.sum() >= 5 else None)
    mm = np.abs(D["depth_h"] - D["depth_k"]) <= 0.05
    res["matched_depth"] = dict(rom(D["h_kd"][mm], D["k_kd"][mm], rng) if mm.sum() >= 5 else dict(n=int(mm.sum())))
    pos = (D["h_kd"] > 0) & (D["k_kd"] > 0)
    rho = spearmanr(np.log2(D["h_kd"][pos] / D["k_kd"][pos]), (D["depth_h"] - D["depth_k"])[pos])
    res["spearman_logratio_vs_depth_difference"] = dict(n=int(pos.sum()), rho=float(rho[0]), p=float(rho[1]))
    res["rows"] = rows
    json.dump(res, open(out + "/metrics_kd_depth.json", "w"), indent=1)
    for k, v in res.items():
        if k not in ("rules", "rows"):
            print(k, json.dumps(v), flush=True)


if __name__ == "__main__":
    main()
