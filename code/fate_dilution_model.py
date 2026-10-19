"""Survivor-knockdown scaling model, test on hESC vs K562 (A100; fate paper C6 prelude; predictions P1-P2 pre-stated in
notes/fate_paper_plan.md §3, commit e81be82). Survivors = knocked-down cells (residual r_kd, effect D) + escapers
(residual 1, effect 0): mean residual R = (1-e) r_kd + e, so e = (R - r_kd) / (1 - r_kd), clipped to [0, 0.95]; mean
effect (1-e) D; the cross-unit magnitude is quadratic, M ~ (1-e_A)(1-e_B) D^2.
hESC magnitude: guide halves (pooled residual for both halves); K562: day-6 x day-8 screens (residual of each screen).
  y = log2(M_hESC / M_K562),  x = 2 log2(1-e_h) - log2(1-e_6) - log2(1-e_8)
P1: Theil-Sen slope of y on x and Spearman (null: x shuffled across targets). P2: paired median ratio before / after
dividing each magnitude by its (1-e) factors, all targets and DepMap < -1. r_kd = median residual of the non-essential
tier per context (main), 0.1 for all (sensitivity). Linear dose-response with uniform partial knockdown gives the same
(1-R) scaling, so this does not separate escaper mixtures from partial knockdown.
Usage: fate_dilution_model.py <samescale_out dir> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
from scipy.stats import theilslopes, spearmanr


def esc(R, rkd):
    return np.clip((R - rkd) / (1 - rkd), 0, 0.95)


def main():
    d, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
    z = np.load(d + "/per_target.npz", allow_pickle=True)
    e = z["essentiality"]; Rh, R6, R8 = z["res_h"], z["res_6"], z["res_8"]
    Mh, Mk = z["h_guide_lfc"], z["k_cross_lfc"]
    ne = np.isfinite(e) & (e > -0.2)
    rk_main = {"h": float(np.nanmedian(Rh[ne])), "k6": float(np.nanmedian(R6[ne])), "k8": float(np.nanmedian(R8[ne]))}
    res = {"rules": __doc__, "r_kd_main": rk_main}
    for label, rk in (("main", rk_main), ("rkd_0.1", {"h": 0.1, "k6": 0.1, "k8": 0.1})):
        eh, e6, e8 = esc(Rh, rk["h"]), esc(R6, rk["k6"]), esc(R8, rk["k8"])
        fh, fk = (1 - eh) ** 2, (1 - e6) * (1 - e8)
        ok = np.isfinite(Rh) & np.isfinite(R6) & np.isfinite(R8) & (Mh > 0) & (Mk > 0)
        y = np.log2(Mh[ok] / Mk[ok]); x = np.log2(fh[ok] / fk[ok])
        sl, ic, lo, hi = theilslopes(y, x)
        rho = spearmanr(x, y)[0]
        rng = np.random.default_rng(0)
        null = [theilslopes(y, rng.permutation(x))[0] for _ in range(200)]
        ess1 = (e[ok] < -1)
        before = float(2 ** np.median(y)); after = float(2 ** np.median(y - x))
        before_e = float(2 ** np.median(y[ess1])); after_e = float(2 ** np.median((y - x)[ess1]))
        # spread of x: how much of the target-to-target variance in y the scaling accounts for
        resid_var = float(np.var(y - sl * x - ic)); tot_var = float(np.var(y))
        res[label] = dict(n=int(ok.sum()), slope=float(sl), slope_ci95=[float(lo), float(hi)], intercept=float(ic),
                          spearman=float(rho), null_slope_p97_5=float(np.percentile(null, 97.5)),
                          paired_median_ratio_before=before, paired_median_ratio_after=after,
                          ess_before=before_e, ess_after=after_e, n_ess=int(ess1.sum()),
                          var_explained=1 - resid_var / tot_var,
                          e_median={"hesc": float(np.median(eh[ok])), "k562_d6": float(np.median(e6[ok])), "k562_d8": float(np.median(e8[ok]))},
                          e_median_ess={"hesc": float(np.median(eh[ok][ess1])), "k562_d6": float(np.median(e6[ok][ess1]))})
        # by x bins
        bins = np.quantile(x, [0, 0.2, 0.4, 0.6, 0.8, 1.0]); bx = np.digitize(x, bins[1:-1])
        res[label]["by_x_quintile"] = [dict(x_median=float(np.median(x[bx == k])), y_ratio=float(2 ** np.median(y[bx == k])), n=int((bx == k).sum()))
                                       for k in range(5)]
        print(label, json.dumps(res[label]), flush=True)
    json.dump(res, open(out + "/metrics_dilution_model.json", "w"), indent=1)


if __name__ == "__main__":
    main()
