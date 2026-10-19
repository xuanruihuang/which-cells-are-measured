"""Fate paper redesign section 3.3 (notes/fate_analysis_redesign.md; expectations L1-L3 and the noise-aware prediction
pre-stated, commits 0829f15 and f958f28). A100.
Per target: measured dilution y = log2(M_all / M_kd), M = cross-signal between the two guide-parity halves (all cells,
or knocked-down cells only); predicted dilution x = log2((1 - e)^2), e = clip((R - r_kd) / (1 - r_kd), 0, 0.95),
R the survivor residual (own-target counts / expected), r_kd the context's non-essential median residual.
Noise in R: x_A, x_B from the two guide-parity halves; reliability of the pooled x = 2 rho / (1 + rho), rho the Pearson
correlation of x_A and x_B across targets; de-attenuated slope = OLS slope of y on pooled x / reliability.
Contexts: hESC (kd-group extraction, thinning-null corrected M_kd), CD4 donors 2-4 resting (budget extraction, guide
unit, no thinning null on the guide unit), K562 day 6 (random halves, point check only).
Null: pooled x permuted across targets within a DepMap tier, 200 times.
Usage: fate_dilution_curve.py <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr, theilslopes
from fate_budget import Ctx, budget, essentiality
import config

DEPMAP = config.get("DEPMAP")
KDG = config.get("FATE_ROOT") + "/kdgroup/metrics_kdgroup.json"
PAN_KDG = config.get("PAN_KDGROUP")
CD4_DIR = config.get("FATE_ROOT") + "/budget_cd4"
K562_DIR = config.get("FATE_ROOT") + "/budget"
R_KD_HESC = 0.218          # non-essential median residual of the hESC per-target table (section 3.1)
MIN_EXP_HALF = 10.0
TIERS = ((-9, -1.0), (-1.0, -0.5), (-0.5, -0.2), (-0.2, 9))


def xform(R, rkd):
    e = np.clip((np.asarray(R, float) - rkd) / (1 - rkd), 0, 0.95)
    return np.log2((1 - e) ** 2)


def tier_of(e):
    out = np.full(len(e), -1)
    for i, (lo, hi) in enumerate(TIERS):
        out[(e >= lo) & (e < hi)] = i
    return out


def fit(x, xa, xb, y, ess, rng, nboot=1000, nperm=200):
    ok = np.isfinite(x) & np.isfinite(xa) & np.isfinite(xb) & np.isfinite(y)
    x, xa, xb, y, ess = x[ok], xa[ok], xb[ok], y[ok], ess[ok]
    n = len(x)
    if n < 20 or np.std(x) == 0:
        return dict(n=int(n))

    def one(ix):
        rho = np.corrcoef(xa[ix], xb[ix])[0, 1]; rel = 2 * rho / (1 + rho)
        b = np.polyfit(x[ix], y[ix], 1)[0]
        return b, rel, b / rel if rel > 0 else np.nan
    b, rel, bd = one(np.arange(n))
    bs = np.array([one(rng.integers(0, n, n))[2] for _ in range(nboot)])
    ts = theilslopes(y, x)[0]
    tr = tier_of(ess); null = []
    for _ in range(nperm):
        xp = x.copy()
        for i in range(len(TIERS)):
            m = np.where(tr == i)[0]; xp[m] = x[rng.permutation(m)]
        null.append(np.polyfit(xp, y, 1)[0] / rel if rel > 0 else np.nan)
    null = np.array(null)
    return dict(n=int(n), rho_halves=float(np.corrcoef(xa, xb)[0, 1]), reliability=float(rel), ols_slope=float(b),
                theil_sen_slope=float(ts), deattenuated_slope=float(bd),
                deattenuated_ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                boot_sd=float(np.nanstd(bs)), spearman=float(spearmanr(x, y)[0]),
                null_q975=float(np.nanpercentile(null, 97.5)), null_median=float(np.nanmedian(null)),
                x_median=float(np.median(x)), y_median=float(np.median(y)),
                frac_x_zero=float((x == 0).mean())), (x, y, ess)


def hesc(ess):
    rows = json.load(open(KDG))["rows"]
    idx = json.load(open(PAN_KDG + "/index.json")); T = idx["targets"]
    C = np.load(PAN_KDG + "/counts.npy").reshape(len(T), 2, 3, 4)
    tpos = {t: k for k, t in enumerate(T)}
    out = {k: [] for k in ("target", "R", "RA", "RB", "y", "ess")}
    for r in rows:
        t = r["target"]; k = tpos[t]
        own = C[k, :, :, 2].sum(1); mu = C[k, :, :, 3].sum(1)
        if mu.min() < MIN_EXP_HALF:
            continue
        ma, mk = r.get("m_all"), r.get("m_kd_corr")
        y = np.log2(ma / mk) if (ma is not None and mk is not None and np.isfinite(ma) and np.isfinite(mk)
                                 and ma > 0 and mk > 0) else np.nan
        for kk, v in (("target", t), ("R", own.sum() / mu.sum()), ("RA", own[0] / mu[0]), ("RB", own[1] / mu[1]),
                      ("y", y), ("ess", ess.get(t, np.nan))):
            out[kk].append(v)
    return {k: np.array(v) for k, v in out.items()}, R_KD_HESC


def budget_ctx(d, lab, unit, ess):
    C = Ctx(d, lab)
    rows = budget(C, ess, unit)
    resid = C.idx["residual"]
    non = [v for t, v in resid.items() if v is not None and np.isfinite(v) and ess.get(t, np.nan) > -0.2]
    rkd = float(np.median(non))
    out = {k: [] for k in ("target", "R", "RA", "RB", "y", "ess")}
    tpos = {t: k for k, t in enumerate(C.tl)}
    for t, r in rows.items():
        cnt = C.cnt[tpos[t]]                                     # (parity, half, phase, group, 4)
        ax = 0 if unit == "guide" else 1
        own = cnt[..., 2].sum(axis=tuple(i for i in range(4) if i != ax))
        mu = cnt[..., 3].sum(axis=tuple(i for i in range(4) if i != ax))
        if mu.min() < MIN_EXP_HALF:
            continue
        mt, mk = r.get("m_total"), r.get("m_kd")
        y = np.log2(mt / mk) if (mk is not None and mt > 0 and mk > 0) else np.nan
        for kk, v in (("target", t), ("R", own.sum() / mu.sum()), ("RA", own[0] / mu[0]), ("RB", own[1] / mu[1]),
                      ("y", y), ("ess", ess.get(t, np.nan))):
            out[kk].append(v)
    return {k: np.array(v) for k, v in out.items()}, rkd


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    ess = essentiality(DEPMAP); rng = np.random.default_rng(31)
    res = {"rules": __doc__}; arrays = {}
    ctxs = [("hESC",) + hesc(ess)]
    for lab in ("CD4_D2rest", "CD4_D3rest", "CD4_D4rest"):
        ctxs.append((lab,) + budget_ctx(CD4_DIR, lab, "guide", ess))
    for name, D, rkd in ctxs:
        x, xa, xb = xform(D["R"], rkd), xform(D["RA"], rkd), xform(D["RB"], rkd)
        r = fit(x, xa, xb, D["y"], D["ess"], rng)
        if isinstance(r, tuple):
            r, (xx, yy, ee) = r
            arrays[name + "_x"], arrays[name + "_y"], arrays[name + "_ess"] = xx, yy, ee
        r["r_kd"] = rkd; r["n_targets_in"] = int(len(D["R"]))
        res[name] = r
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    # L2: heterogeneity of de-attenuated slopes (inverse-variance, bootstrap SDs)
    names = [n for n, *_ in ctxs if "deattenuated_slope" in res[n]]
    b = np.array([res[n]["deattenuated_slope"] for n in names]); s = np.array([res[n]["boot_sd"] for n in names])
    w = 1 / s ** 2; bbar = float((w * b).sum() / w.sum()); Q = float((w * (b - bbar) ** 2).sum())
    from scipy.stats import chi2
    res["L2"] = dict(contexts=names, pooled_slope=bbar, Q=Q, df=len(names) - 1, p=float(chi2.sf(Q, len(names) - 1)))
    # L3: pooled Spearman
    xs = np.concatenate([arrays[n + "_x"] for n in names]); ys = np.concatenate([arrays[n + "_y"] for n in names])
    res["L3_pooled_spearman"] = float(spearmanr(xs, ys)[0]); res["L3_n"] = int(len(xs))
    # K562 point check (random halves, most essential tier)
    Dk, rk = budget_ctx(K562_DIR, "K562", "random", ess)
    xk = xform(Dk["R"], rk); m = np.isfinite(Dk["y"]) & (Dk["ess"] < -1)
    res["K562_point"] = dict(r_kd=rk, n=int(m.sum()), x_median=float(np.median(xk[m])), y_median=float(np.median(Dk["y"][m])))
    print("L2", json.dumps(res["L2"]), "L3", round(res["L3_pooled_spearman"], 3), "K562", json.dumps(res["K562_point"]), flush=True)
    json.dump(res, open(out + "/metrics_dilution_curve.json", "w"), indent=1)
    np.savez(out + "/dilution_curve_points.npz", **arrays)


if __name__ == "__main__":
    main()
