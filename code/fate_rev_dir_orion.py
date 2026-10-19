"""Fate v3r2 external review, checks V13 and V14 (expectations pre-stated in notes/fate_analysis_redesign.md §7.2, commit
f9d9a81). A100.
V13 direction: per-target rho with all cells and with silenced cells (fate_c34_agree.py rows); median change with a
    bootstrap interval over targets (2,000), per stratum and over all strata; the size-matched random-subset null as
    its 2.5th and 97.5th percentiles of the median change (each of the 200 draws gives one median over the fixed targets).
V14 X-Atlas/Orion: in the most essential stratum, the HEK293T / HCT116 ratio on all targets and on targets whose
    residuals differ by at most 0.1, and log2 of their quotient, from the same bootstrap draws (2,000; fate_orion_matched2
    estimator: median of per-target log2 ratios on targets positive on both sides).
Usage: fate_rev_dir_orion.py <out json>. Exploratory."""
import sys, json
import numpy as np
import config

FI = config.get("FATE_ROOT")
TIERS = (("lt_-1", -9, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, 9), ("all", -99, 99))


def main():
    out = sys.argv[1]; rng = np.random.default_rng(13); res = {"rules": __doc__, "V13": {}}
    z = np.load(FI + "/c34/c34_rows.npz", allow_pickle=True)
    E, ra, rk, rr = z["essentiality"], z["rho_all"], z["rho_kd"], z["rho_rand"]
    for nm, lo, hi in TIERS:
        m = np.isfinite(E) & (E >= lo) & (E < hi) & np.isfinite(ra) & np.isfinite(rk)
        if m.sum() < 5:
            continue
        d = rk[m] - ra[m]; n = len(d)
        bs = [np.median(d[rng.integers(0, n, n)]) for _ in range(2000)]
        nul = np.nanmedian(rr[m] - ra[m][:, None], axis=0)
        res["V13"][nm] = dict(n=int(n), median_rho_all=float(np.median(ra[m])), median_rho_kd=float(np.median(rk[m])),
                              median_change=float(np.median(d)), change_of_medians=float(np.median(rk[m]) - np.median(ra[m])),
                              ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                              null_q025=float(np.percentile(nul, 2.5)), null_q975=float(np.percentile(nul, 97.5)), null_median=float(np.median(nul)))
        print("V13", nm, json.dumps(res["V13"][nm]), flush=True)
    z = np.load(FI + "/orion_matched/per_target.npz", allow_pickle=True)
    mag, resid, ess, usable = z["mag"], z["resid"], z["essentiality"], z["usable"]
    both = usable[0] & usable[1] & np.isfinite(resid[0]) & np.isfinite(resid[1]) & np.isfinite(ess) & (ess < -1)
    dR = resid[1] - resid[0]
    ok = both & (mag[0] > 0) & (mag[1] > 0)
    lr = np.log2(mag[1] / np.where(mag[0] > 0, mag[0], np.nan)); eq = np.abs(dR) <= 0.1
    idx = np.where(ok)[0]; n = len(idx)
    def stat(ix):
        a = lr[ix]; q = eq[ix]
        return np.median(a), np.median(a[q]) if q.sum() >= 3 else np.nan
    p = stat(idx); B = np.array([stat(idx[rng.integers(0, n, n)]) for _ in range(2000)])
    diff = B[:, 1] - B[:, 0]
    res["V14"] = dict(n_all=int(n), n_equal=int(eq[idx].sum()), ratio_all=float(2 ** p[0]), ratio_equal=float(2 ** p[1]),
                      log2_quotient=float(p[1] - p[0]), log2_quotient_ci95=[float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))],
                      ratio_all_ci95=[float(2 ** np.nanpercentile(B[:, 0], 2.5)), float(2 ** np.nanpercentile(B[:, 0], 97.5))],
                      ratio_equal_ci95=[float(2 ** np.nanpercentile(B[:, 1], 2.5)), float(2 ** np.nanpercentile(B[:, 1], 97.5))])
    print("V14", json.dumps(res["V14"]), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
