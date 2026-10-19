"""Fate v3r2 external review, checks V3, V5, V15 (expectations pre-stated in notes/fate_analysis_redesign.md §7.2,
commit f9d9a81). A100.
V3  per target, measured dilution y = log2(M_all / M_kd) against x = log2((1 - e_mix)^2), e_mix the NB mixture weight
    of fate_mixture.py (clipped at 0.95), and against x_R from the mean residual (as Fig. 6a), on the same targets.
    hESC: M from fate_kdgroup.py rows (guide halves; M_kd thinning-null corrected); e_mix pooled, with the two guide-
    parity fits of fate_rev_mixture.py V8 as the two noisy measurements (reliability 2 rho / (1 + rho)).
    K562: M from fate_budget_screens.py rows (two screens); e_mix of the two screens' fits as the two measurements.
    RPE1: M from fate_budget.py rows (random halves; thinning-null corrected); one fit, OLS and Theil-Sen only.
    Also: e_mix as fitted (primary) and BIC-selected (e = 0 when the mixture is not preferred).
V5  CD4 donors 2-4 resting, guide units: knocked-down cells of guide side u against the thinning-null cells of random
    half u (both relative to their own reference, i.e. log2(P_kd,u / P_null,u)), cross-signal across u; measured
    dilution and the residual-to-dilution fit recomputed.
V15 hESC Fig. 6a fit without the targets whose residual-based e is clipped at 0.
Usage: fate_rev_dilution.py <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr, theilslopes
import config
from fate_budget import Ctx, essentiality, vectors, response_genes, MIN_UNIT
import fate_dilution_curve as DC

FI = config.get("FATE_ROOT")


def xe(e):
    return np.log2((1 - np.clip(np.asarray(e, float), 0, 0.95)) ** 2)


def simple(x, y):
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 10 or np.std(x) == 0:
        return dict(n=int(len(x)))
    b, a = np.polyfit(x, y, 1); ts = theilslopes(y, x)
    r2 = 1 - np.var(y - (a + b * x)) / np.var(y)
    return dict(n=int(len(x)), ols_slope=float(b), ols_intercept=float(a), theil_sen_slope=float(ts[0]), theil_sen_ci=[float(ts[2]), float(ts[3])],
                spearman=float(spearmanr(x, y)[0]), r2=float(r2), median_offset=float(np.median(y - x)), x_median=float(np.median(x)), y_median=float(np.median(y)))


def mixrows():
    m1 = json.load(open(FI + "/mixture/metrics_mixture.json"))["rows"]; m2 = json.load(open(FI + "/mixture_more/metrics_mixture_more.json"))["rows"]
    f = lambda rr: {r["target"]: r for r in rr}
    return dict(hESC=f(m1["hESC"]), K562_d8=f(m1["K562_gw"]), K562_d6=f(m2["K562ess"]), RPE1=f(m2["RPE1"]))


def e_of(r, bic):
    if r is None:
        return np.nan
    if bic and r["nb"]["dbic"] <= 10:
        return 0.0
    return r["nb"]["e"]


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); rng = np.random.default_rng(31)
    ess = essentiality(DC.DEPMAP); M = mixrows(); res = {"rules": __doc__, "V3": {}}; pts = {}
    # ---- hESC
    D, rkd = DC.hesc(ess)
    v8 = {x["target"]: x for x in json.load(open(FI + "/rev_mixture/metrics_rev_mixture.json"))["V8_rows"]}
    for bic in (False, True):
        lab = "bic_selected" if bic else "fitted"
        e = np.array([e_of(M["hESC"].get(t), bic) for t in D["target"]])
        ea = np.array([np.nan if (t not in v8 or v8[t]["p0"] is None) else (0.0 if bic and v8[t]["p0"]["dbic"] <= 10 else v8[t]["p0"]["e"]) for t in D["target"]])
        eb = np.array([np.nan if (t not in v8 or v8[t]["p1"] is None) else (0.0 if bic and v8[t]["p1"]["dbic"] <= 10 else v8[t]["p1"]["e"]) for t in D["target"]])
        x, xa, xb = xe(e), xe(ea), xe(eb); y = D["y"]
        r = DC.fit(x, xa, xb, y, D["ess"], rng)
        r = r[0] if isinstance(r, tuple) else r
        ok = np.isfinite(x) & np.isfinite(xa) & np.isfinite(xb) & np.isfinite(y)
        xR = DC.xform(D["R"], rkd)
        pts["hESC_%s_x" % lab], pts["hESC_%s_y" % lab], pts["hESC_%s_xR" % lab] = x[ok], y[ok], xR[ok]
        res["V3"]["hESC_" + lab] = dict(deattenuated=r, simple_emix=simple(x[ok], y[ok]), simple_eR_same_targets=simple(xR[ok], y[ok]),
                                        simple_emix_all_targets=simple(x, y))
        print("V3 hESC", lab, json.dumps(res["V3"]["hESC_" + lab]), flush=True)
    # V15
    xR, xRa, xRb = DC.xform(D["R"], rkd), DC.xform(D["RA"], rkd), DC.xform(D["RB"], rkd)
    keep = xR < 0
    r = DC.fit(xR[keep], xRa[keep], xRb[keep], D["y"][keep], D["ess"][keep], rng); r = r[0] if isinstance(r, tuple) else r
    r_all = DC.fit(xR, xRa, xRb, D["y"], D["ess"], rng); r_all = r_all[0] if isinstance(r_all, tuple) else r_all
    res["V15_hESC"] = dict(without_clipped=r, with_clipped=r_all, n_clipped=int((~keep & np.isfinite(D["y"])).sum()))
    print("V15", json.dumps(res["V15_hESC"]), flush=True)
    # ---- K562
    rows = json.load(open(FI + "/budget_d8/metrics_budget_screens.json"))["rows"]
    T = [t for t, r in rows.items() if "m_kd" in r]
    mt = np.array([rows[t]["m_total"] for t in T]); mk = np.array([rows[t]["m_kd"] for t in T]); E = np.array([rows[t]["essentiality"] for t in T])
    y = np.where((mt > 0) & (mk > 0), np.log2(np.maximum(mt, 1e-12) / np.maximum(mk, 1e-12)), np.nan)
    for bic in (False, True):
        lab = "bic_selected" if bic else "fitted"
        ea = np.array([e_of(M["K562_d6"].get(t), bic) for t in T]); eb = np.array([e_of(M["K562_d8"].get(t), bic) for t in T])
        xa, xb = xe(ea), xe(eb); x = (xa + xb) / 2
        r = DC.fit(x, xa, xb, y, E, rng); r = r[0] if isinstance(r, tuple) else r
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(xa) & np.isfinite(xb)
        pts["K562_%s_x" % lab], pts["K562_%s_y" % lab] = x[ok], y[ok]
        res["V3"]["K562_" + lab] = dict(deattenuated=r, simple_emix=simple(x[ok], y[ok]))
        print("V3 K562", lab, json.dumps(res["V3"]["K562_" + lab]), flush=True)
    # ---- RPE1 (random halves)
    b = json.load(open(FI + "/budget/metrics_fate_budget.json"))["RPE1"]["rows_random"]
    T = [t for t, r in b.items() if "m_kd" in r and t in M["RPE1"]]
    mt = np.array([b[t]["m_total"] for t in T]); mk = np.array([b[t]["m_kd"] for t in T])
    y = np.where((mt > 0) & (mk > 0), np.log2(np.maximum(mt, 1e-12) / np.maximum(mk, 1e-12)), np.nan)
    for bic in (False, True):
        lab = "bic_selected" if bic else "fitted"
        x = xe([e_of(M["RPE1"].get(t), bic) for t in T])
        okr = np.isfinite(x) & np.isfinite(y); pts["RPE1_%s_x" % lab], pts["RPE1_%s_y" % lab] = x[okr], y[okr]
        res["V3"]["RPE1_" + lab] = dict(simple_emix=simple(x, y))
        print("V3 RPE1", lab, json.dumps(res["V3"]["RPE1_" + lab]), flush=True)
    # ---- V5 CD4 with the thinning null on guide units
    V5 = {}
    for lab in ("CD4_D2rest", "CD4_D3rest", "CD4_D4rest"):
        C = Ctx(FI + "/budget_cd4", lab); sel = response_genes(C); newpos = {q: j for j, q in enumerate(sel)}
        non = [v for t, v in C.idx["residual"].items() if v is not None and np.isfinite(v) and ess.get(t, np.nan) > -0.2]
        rk = float(np.median(non)); out_r = {k: [] for k in ("R", "RA", "RB", "y_old", "y_new", "ess")}
        for k, t in enumerate(C.tl[:-1]):
            if C.gpt.get(t, 1) < 2:
                continue
            V = [vectors(C, k, "guide", u, [0, 1, 2], sel) for u in (0, 1)]; K = [vectors(C, k, "guide", u, [0], sel) for u in (0, 1)]
            if any(v is None for v in V + K):
                continue
            nl = []
            for u in (0, 1):
                if C.ncnt[k, :, u, 0].sum() < MIN_UNIT:
                    nl.append(None); continue
                A = np.asarray(C.nraw[k, :, u]).sum(0)[sel]; lib = C.ncnt[k, :, u, 1].sum()
                nl.append(np.log2((A + 0.5) / lib))
            if any(v is None for v in nl):
                continue
            j = newpos.get(C.gq.get(t, -1))
            Ea = [V[0][0].copy(), V[1][0].copy()]; Ek = [K[0][0].copy(), K[1][0].copy()]
            # silenced cells of guide side u vs null cells of random half u: log2(P_kd,u) - log2(P_null,u)
            Akd = [C.sums(k, "guide", u, [0]) for u in (0, 1)]
            Ekn = [np.log2((Akd[u][0][:, sel].sum(0) + 0.5) / Akd[u][1].sum()) - nl[u] for u in (0, 1)]
            for v in Ea + Ek + Ekn:
                if j is not None:
                    v[j] = 0
            ma, mko, mkn = float(Ea[0] @ Ea[1]), float(Ek[0] @ Ek[1]), float(Ekn[0] @ Ekn[1])
            cnt = C.cnt[k]; own = cnt[..., 2].sum(axis=(1, 2, 3)); mu = cnt[..., 3].sum(axis=(1, 2, 3))
            if mu.min() < DC.MIN_EXP_HALF:
                continue
            for kk, v in (("R", own.sum() / mu.sum()), ("RA", own[0] / mu[0]), ("RB", own[1] / mu[1]), ("ess", ess.get(t, np.nan)),
                          ("y_old", np.log2(ma / mko) if ma > 0 and mko > 0 else np.nan), ("y_new", np.log2(ma / mkn) if ma > 0 and mkn > 0 else np.nan)):
                out_r[kk].append(v)
        Dd = {k: np.array(v, float) for k, v in out_r.items()}
        x, xa, xb = DC.xform(Dd["R"], rk), DC.xform(Dd["RA"], rk), DC.xform(Dd["RB"], rk)
        pts[lab + "_x"], pts[lab + "_y_old"], pts[lab + "_y_new"] = x, Dd["y_old"], Dd["y_new"]
        f_old = DC.fit(x, xa, xb, Dd["y_old"], Dd["ess"], rng); f_old = f_old[0] if isinstance(f_old, tuple) else f_old
        f_new = DC.fit(x, xa, xb, Dd["y_new"], Dd["ess"], rng); f_new = f_new[0] if isinstance(f_new, tuple) else f_new
        V5[lab] = dict(r_kd=rk, n=int(len(x)), median_y_old=float(np.nanmedian(Dd["y_old"])), median_y_new=float(np.nanmedian(Dd["y_new"])),
                       median_x=float(np.median(x)), n_y_old=int(np.isfinite(Dd["y_old"]).sum()), n_y_new=int(np.isfinite(Dd["y_new"]).sum()),
                       fit_old=f_old, fit_new=f_new)
        print("V5", lab, json.dumps(V5[lab]), flush=True)
    res["V5"] = V5
    np.savez(out + "/rev_dilution_points.npz", **pts)
    json.dump(res, open(out + "/metrics_rev_dilution.json", "w"), indent=1, default=lambda v: None if v is None else float(v))


if __name__ == "__main__":
    main()
