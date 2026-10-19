"""R1 supplement (fate paper §8; expectation O2' pre-stated in notes/fate_paper_plan.md §8 after the first round, before
running): HEK293T / HCT116 magnitude ratio with knockdown matched as EQUAL residual on both sides, not low residual.
Input: fate_orion_matched.py per_target.npz (min_resp 0.5: mag, resid, n_cells, usable, inert, expressed, essentiality).
Per tier: paired ratio on |R_HEK - R_HCT| <= 0.1 (and <= 0.05); ratio by bins of dR = R_HEK - R_HCT; model-corrected
ratio, each magnitude divided by (1 - e)^2 with e = (R - rkd) / (1 - rkd), rkd the line's non-essential median residual,
clipped to [0, 0.95]; discernible fractions (above the inert 97.5th percentile) on the |dR| <= 0.1 set; inert floor.
Usage: fate_orion_matched2.py <orion_matched dir> <out json>. Exploratory."""
import sys, json
import numpy as np

TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf))
BINS = (("dR<-0.1", -np.inf, -0.1), ("|dR|<=0.1", -0.1, 0.1), ("0.1<dR<=0.3", 0.1, 0.3), ("dR>0.3", 0.3, np.inf))
NBOOT = 1000
LINES = ("HCT116", "HEK293T")


def paired_ratio(a, b, rng):
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if ok.sum() < 5:
        return dict(n=int(ok.sum()), ratio=None, ci95=None)
    lr = np.log2(a[ok] / b[ok]); n = len(lr)
    bs = np.array([np.median(lr[rng.integers(0, n, n)]) for _ in range(NBOOT)])
    return dict(n=int(n), ratio=float(2 ** np.median(lr)), ci95=[float(2 ** np.percentile(bs, 2.5)), float(2 ** np.percentile(bs, 97.5))])


def frac_ci(x, rng):
    x = np.asarray(x, bool); n = len(x)
    if n < 5:
        return dict(n=int(n), frac=None, ci95=None)
    bs = np.array([x[rng.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return dict(n=int(n), frac=float(x.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def main():
    d, outf = sys.argv[1:3]
    z = np.load(d + "/per_target.npz", allow_pickle=True)
    mag, resid, ess, usable, inert, expressed = z["mag"], z["resid"], z["essentiality"], z["usable"], z["inert"], z["expressed"]
    both = usable[0] & usable[1] & np.isfinite(resid[0]) & np.isfinite(resid[1])
    rng = np.random.default_rng(1)
    noness = both & (ess > -0.2)
    rkd = [float(np.nanmedian(resid[li, noness])) for li in range(2)]
    e = [np.clip((resid[li] - rkd[li]) / (1 - rkd[li]), 0, 0.95) for li in range(2)]
    corr = [mag[li] / (1 - e[li]) ** 2 for li in range(2)]
    thr = [float(np.nanpercentile(mag[li, usable[0] & usable[1] & inert[li]], 97.5)) for li in range(2)]
    floor = {L: dict(inert_median=float(np.nanmedian(mag[li, usable[0] & usable[1] & inert[li]])), inert_p97_5=thr[li],
                     nonessential_median=float(np.nanmedian(mag[li, noness]))) for li, L in enumerate(LINES)}
    dR = resid[1] - resid[0]
    res = {"rules": __doc__, "rkd_nonessential_median": dict(zip(LINES, rkd)), "floor": floor, "tiers": {}}
    print("rkd", rkd, "floor", json.dumps(floor), flush=True)
    for nm, lo, hi in TIERS:
        tm = both & np.isfinite(ess) & (ess >= lo) & (ess < hi)
        row = {"n": int(tm.sum()), "all": paired_ratio(mag[1, tm], mag[0, tm], rng),
               "equal_resid_0.1": paired_ratio(mag[1, tm & (np.abs(dR) <= 0.1)], mag[0, tm & (np.abs(dR) <= 0.1)], rng),
               "equal_resid_0.05": paired_ratio(mag[1, tm & (np.abs(dR) <= 0.05)], mag[0, tm & (np.abs(dR) <= 0.05)], rng),
               "model_corrected_all": paired_ratio(corr[1][tm], corr[0][tm], rng),
               "median_resid": {L: float(np.nanmedian(resid[li, tm])) for li, L in enumerate(LINES)},
               "bins": {}}
        for bn, blo, bhi in BINS:
            bm = tm & (dR > blo) & (dR <= bhi)
            row["bins"][bn] = paired_ratio(mag[1, bm], mag[0, bm], rng)
            row["bins"][bn]["median_resid"] = {L: float(np.nanmedian(resid[li, bm])) if bm.any() else None for li, L in enumerate(LINES)}
        ex = tm & expressed[0] & expressed[1]
        row["discernible"] = {L: dict(all=frac_ci(mag[li, ex] > thr[li], rng), equal_resid_0_1=frac_ci(mag[li, ex & (np.abs(dR) <= 0.1)] > thr[li], rng))
                              for li, L in enumerate(LINES)}
        res["tiers"][nm] = row
        print("== tier", nm, "n", row["n"], "resid", json.dumps(row["median_resid"]), flush=True)
        for k in ("all", "equal_resid_0.1", "equal_resid_0.05", "model_corrected_all"):
            print("   %-20s %s" % (k, json.dumps(row[k])), flush=True)
        for bn, v in row["bins"].items():
            print("   bin %-12s %s" % (bn, json.dumps(v)), flush=True)
        print("   discernible", json.dumps(row["discernible"]), flush=True)
    json.dump(res, open(outf, "w"), indent=1)


if __name__ == "__main__":
    main()
