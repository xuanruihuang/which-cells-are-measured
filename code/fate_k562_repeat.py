"""R5 (fate paper §8; expectations K1-K3 pre-stated in notes/fate_paper_plan.md §8): the residual-to-dilution slope
(§2.6) and the silenced-cells-keep-direction test (§2.7), repeated in K562 with the two independent screens (day-6
essential library, day-8 genome-wide library) as the units.
Dilution curve: per target, y = log2(M_total / M_kd) from fate_budget_screens.py rows (cross-screen cross-signals,
M_kd from silenced cells with each screen's thinning null subtracted); x = log2(1 - e)^2 with e from the mean of the two
screens' residuals (each screen's r_kd = its non-essential median residual); x_A, x_B from the two screens; fit as
fate_dilution_curve.fit (reliability 2 rho / (1 + rho), de-attenuated OLS slope, bootstrap CI, within-tier permutation
null). Direction: de-attenuated correlation with hESC (guide-parity units) of the K562 effect, computed from all cells
and from silenced cells (per-screen pooled vectors as fate_budget_screens.pooled / pooled_null), same targets.
Usage: fate_k562_repeat.py <depmap csv> <budget dir d6> <budget dir d8> <budget_screens json> <vectors dir> <out json>."""
import sys, json
import numpy as np
from fate_budget import Ctx, essentiality, control_cp10k, ALL, MIN_RESP_CP10K
from fate_budget_screens import pooled, pooled_null
from fate_dilution_curve import fit, xform

TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf), ("pooled", -np.inf, np.inf))
NBOOT = 1000


def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def rkd_of(C, ess):
    non = [v for t, v in C.idx["residual"].items() if v is not None and np.isfinite(v) and ess.get(t, np.nan) > -0.2]
    return float(np.median(non))


def main():
    dep, d6, d8, bsj, vdir, outf = sys.argv[1:7]; ess = essentiality(dep); rng = np.random.default_rng(31)
    rows = json.load(open(bsj))["rows"]
    A, B = Ctx(d6, "K562"), Ctx(d8, "K562d8"); rkA, rkB = rkd_of(A, ess), rkd_of(B, ess)
    res = {"rules": __doc__, "r_kd": {"d6": rkA, "d8": rkB}}
    # ---- dilution curve
    T = [t for t, r in rows.items() if "m_kd" in r and np.isfinite(r["resid_d6"]) and np.isfinite(r["resid_d8"])]
    R6 = np.array([rows[t]["resid_d6"] for t in T]); R8 = np.array([rows[t]["resid_d8"] for t in T])
    mt = np.array([rows[t]["m_total"] for t in T]); mk = np.array([rows[t]["m_kd"] for t in T]); E = np.array([rows[t]["essentiality"] for t in T])
    y = np.where((mt > 0) & (mk > 0), np.log2(np.maximum(mt, 1e-12) / np.maximum(mk, 1e-12)), np.nan)
    xa, xb = xform(R6, rkA), xform(R8, rkB); x = (xa + xb) / 2
    ok = np.isfinite(E)
    res["premise_residual_corr_d6_d8"] = dict(pearson=float(np.corrcoef(R6, R8)[0, 1]), n=int(len(R6)),
                                              spearman=float(np.corrcoef(np.argsort(np.argsort(R6)), np.argsort(np.argsort(R8)))[0, 1]))
    r = fit(x[ok], xa[ok], xb[ok], y[ok], E[ok], rng)
    if isinstance(r, tuple):
        r = r[0]
    res["dilution_curve"] = r
    res["dilution_by_tier"] = {}
    for nm, lo, hi in TIERS:
        m = ok & (E >= lo) & (E < hi) & np.isfinite(y)
        res["dilution_by_tier"][nm] = dict(n=int(m.sum()), x_median=med(x[m]), y_median=med(y[m]), offset=(med(y[m]) - med(x[m])) if m.sum() else None,
                                           resid_d6=med(R6[m]), resid_d8=med(R8[m]))
    print("premise", json.dumps(res["premise_residual_corr_d6_d8"]), flush=True)
    print("curve", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    for nm, v in res["dilution_by_tier"].items():
        print("  tier", nm, json.dumps(v), flush=True)
    # ---- direction vs hESC
    z = np.load(vdir + "/hESC_vectors.npz", allow_pickle=True); hg = [str(g) for g in z["genes"]]; ht = [str(t) for t in z["targets"]]
    htot = z["tot_g"]; hpos = {g: j for j, g in enumerate(hg)}; htpos = {t: k for k, t in enumerate(ht)}
    cpA, cpB = control_cp10k(A), control_cp10k(B)
    gA = {g: q for q, g in enumerate(A.genes)}; gB = {g: q for q, g in enumerate(B.genes)}
    shared = [g for g in A.genes if g in gB and g in hpos and cpA[gA[g]] >= MIN_RESP_CP10K and cpB[gB[g]] >= MIN_RESP_CP10K]
    selA = np.array([gA[g] for g in shared]); selB = np.array([gB[g] for g in shared]); selH = np.array([hpos[g] for g in shared])
    posA = {q: j for j, q in enumerate(selA)}; posB = {q: j for j, q in enumerate(selB)}
    tA = {t: k for k, t in enumerate(A.tl)}; tB = {t: k for k, t in enumerate(B.tl)}
    drows = {}
    for t in A.tl[:-1]:
        if t not in tB or t not in htpos or not np.isfinite(htot[htpos[t], 0, 0]):
            continue
        ka, kb = tA[t], tB[t]
        va, vb = pooled(A, ka, ALL, selA), pooled(B, kb, ALL, selB)
        ka_kd, kb_kd = pooled(A, ka, [0], selA), pooled(B, kb, [0], selB)
        nla, nlb = pooled_null(A, ka, selA), pooled_null(B, kb, selB)
        if any(v is None for v in (va, vb, ka_kd, kb_kd, nla, nlb)):
            continue
        Ea, Eb = va[0], vb[0]; Eka, Ekb = ka_kd[0] - nla, kb_kd[0] - nlb
        ja, jb = posA.get(gA.get(t, -1)), posB.get(gB.get(t, -1))
        for v, j in ((Ea, ja), (Eka, ja), (Eb, jb), (Ekb, jb)):
            if j is not None:
                v[j] = 0
        H = htot[htpos[t]][:, selH]; MH = float(H[0] @ H[1])
        Mall, Mkd = float(Ea @ Eb), float(Eka @ Ekb)
        if MH <= 0 or Mall <= 0 or Mkd <= 0:
            continue
        hm = H.mean(0)
        drows[t] = dict(essentiality=float(ess.get(t, np.nan)), rho_all=float(((Ea + Eb) / 2 @ hm) / np.sqrt(Mall * MH)),
                        rho_kd=float(((Eka + Ekb) / 2 @ hm) / np.sqrt(Mkd * MH)), n_kd=int(ka_kd[2] + kb_kd[2]), n_all=int(va[2] + vb[2]))
    res["direction"] = {"n_shared_genes": len(shared), "n_targets": len(drows), "tiers": {}}
    for nm, lo, hi in TIERS:
        ts = [t for t, r in drows.items() if np.isfinite(r["essentiality"]) and lo <= r["essentiality"] < hi]
        if len(ts) < 5:
            res["direction"]["tiers"][nm] = dict(n=len(ts)); continue
        ra = np.array([drows[t]["rho_all"] for t in ts]); rk = np.array([drows[t]["rho_kd"] for t in ts]); d = rk - ra; n = len(d)
        bs = np.array([np.median(d[rng.integers(0, n, n)]) for _ in range(NBOOT)])
        res["direction"]["tiers"][nm] = dict(n=n, rho_all_median=float(np.median(ra)), rho_kd_median=float(np.median(rk)), delta_median=float(np.median(d)),
                                             delta_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], frac_kd_higher=float(np.mean(d > 0)),
                                             n_kd_cells_median=med([drows[t]["n_kd"] for t in ts]), n_all_cells_median=med([drows[t]["n_all"] for t in ts]))
        print("  direction", nm, json.dumps(res["direction"]["tiers"][nm]), flush=True)
    res["rows"] = {"dilution": {t: dict(x=float(x[i]), y=float(y[i]), ess=float(E[i])) for i, t in enumerate(T)}, "direction": drows}
    json.dump(res, open(outf, "w"), indent=1)


if __name__ == "__main__":
    main()
