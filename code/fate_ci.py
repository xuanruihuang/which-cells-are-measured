"""Intervals for the fate paper's headline numbers (2026-10-18; for notes/fate_paper_ledger.md).
Three kinds of uncertainty are kept apart, never merged into one interval:
  within   target bootstrap (B draws, targets resampled with replacement, everything else fixed): the uncertainty
           from which targets happen to be in a tier, inside one experiment
  replicate spread over biological replicates where >= 2 exist (hESC: the two guides estimated separately;
           CD4: donors D2 / D3 / D4; K562: the two independent screens) - reported as the set of values, no interval
  design   alternative unit definitions or thresholds, already in the findings (copied into the ledger, not here)
With two units per context a between-unit variance cannot be estimated; that limit is stated in the ledger once.
Usage: fate_ci.py <depmap csv> <out json>. Reads the outputs under FATE_ROOT (config.py). Exploratory."""
import sys, json
import numpy as np
import config
from fate_crosscontext import essentiality, load, units, TIERS

B = 2000
F = config.get("FATE_ROOT")
rng = np.random.default_rng(2026)


def tier_of(e):
    for nm, lo, hi in TIERS:
        if np.isfinite(e) and lo <= e < hi:
            return nm
    return None


def pmed_ratio(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return float(2 ** np.median(np.log2(a[ok] / b[ok]))) if ok.sum() else np.nan


def share(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float); ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    return float(np.median(num[ok] / den[ok])) if ok.sum() else np.nan


def boot(fn, *arrs):
    """Point estimate and 95% percentile interval of fn over target-resampled copies of the parallel arrays."""
    arrs = [np.asarray(a, float) for a in arrs]; n = len(arrs[0])
    est = fn(*arrs)
    if n < 5:
        return dict(est=est, ci=None, n=n)
    bs = [fn(*[a[i] for a in arrs]) for i in (rng.integers(0, n, n) for _ in range(B))]
    bs = np.array(bs, float); bs = bs[np.isfinite(bs)]
    return dict(est=est, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=n)


def main():
    dep, outf = sys.argv[1:3]; ess = essentiality(dep); out = {"rules": __doc__}

    # bb: hESC / K562 same-scale (guide halves x cross-screen), all targets and both-sides good knockdown
    z = np.load(F + "/samescale_out/per_target.npz", allow_pickle=True)
    h, k = z["h_guide_lfc"], z["k_cross_lfc"]; good = (z["res_h"] <= 0.3) & (z["res_6"] <= 0.3) & (z["res_8"] <= 0.3)
    out["bb"] = dict(all_targets=boot(pmed_ratio, h, k), E2_both_good=boot(pmed_ratio, h[good], k[good]),
                     E2_random_design=boot(pmed_ratio, z["h_random_lfc"][good], z["k_random6_lfc"][good]))

    # bc: hESC library-normalised abundance per tier; the two guides separately (replicate)
    rows = [r for r in json.load(open(F + "/pan_libnorm/pan_library_norm.json"))["rows"] if np.isfinite(r.get("norm", np.nan))]
    out["bc"] = {}
    for nm, lo, hi in TIERS:
        rs = [r for r in rows if np.isfinite(r["essentiality"]) and lo <= r["essentiality"] < hi]
        g = [[r["parity_norm"][j] for r in rs if len(r["parity_norm"]) > j and r["parity_norm"][j] is not None and np.isfinite(r["parity_norm"][j])] for j in (0, 1)]
        out["bc"][nm] = dict(norm=boot(lambda v: float(np.median(v)), [r["norm"] for r in rs]),
                             raw=boot(lambda v: float(np.median(v)), [r["raw"] for r in rs]),
                             replicate_guide_medians=[float(np.median(x)) if x else None for x in g])

    # bj: RPE1 vs K562 composition decomposition (random halves of one screen each)
    P = json.load(open(F + "/phase_decomp/metrics_phase_decomp.json"))["rows"]
    out["bj"] = {}
    for nm, lo, hi in TIERS:
        d = {}
        for lab in ("RPE1", "K562"):
            R = P[lab]; ts = [t for t in R if tier_of(ess.get(t, np.nan)) == nm]
            d[lab + "_comp_share"] = boot(share, [R[t]["m_comp"] for t in ts], [R[t]["m_total"] for t in ts])
        sh = [t for t in P["RPE1"] if t in P["K562"] and tier_of(ess.get(t, np.nan)) == nm]
        d["S2_total_ratio"] = boot(pmed_ratio, [P["RPE1"][t]["m_total"] for t in sh], [P["K562"][t]["m_total"] for t in sh])
        d["S2_within_ratio"] = boot(pmed_ratio, [P["RPE1"][t]["m_within"] for t in sh], [P["K562"][t]["m_within"] for t in sh])
        out["bj"][nm] = d

    # bm / bn: unified budget (random halves), composition share and selection term per context
    out["bm"] = {}
    srcs = [(F + "/budget/metrics_fate_budget.json", ("K562", "RPE1", "hESC")),
            (F + "/budget_cd4/nc/metrics_fate_budget.json", ("CD4_D2rest",)),
            (F + "/budget_cd4/d34/metrics_fate_budget.json", ("CD4_D3rest", "CD4_D4rest"))]
    for path, labs in srcs:
        M = json.load(open(path))
        for lab in labs:
            R = M[lab]["rows_random"]; d = {}
            for nm, lo, hi in TIERS:
                ts = [t for t, r in R.items() if tier_of(r["essentiality"]) == nm]
                mt = [R[t]["m_total"] for t in ts]; mc = [R[t]["m_comp"] for t in ts]
                mk = [R[t].get("m_kd", np.nan) for t in ts]
                d[nm] = dict(comp_share=boot(share, mc, mt), selection=boot(share, mk, mt),
                             m_total_median=boot(lambda v: float(np.median(v)), mt))
            out["bm"][lab] = d

    # bv: K562 across the two independent screens
    S = json.load(open(F + "/budget_d8/metrics_budget_screens.json"))["rows"]; out["bv"] = {}
    for nm, lo, hi in TIERS:
        ts = [t for t, r in S.items() if tier_of(r["essentiality"]) == nm]
        out["bv"][nm] = dict(comp_share=boot(share, [S[t]["m_comp"] for t in ts], [S[t]["m_total"] for t in ts]),
                             selection=boot(share, [S[t].get("m_kd", np.nan) for t in ts], [S[t]["m_total"] for t in ts]))

    # bo / br: per-target delta rho against K562 (own composition; same-tier mismatch, 100 donors)
    d = F + "/xctx"; Rk = load(d, "K562"); out["br"] = {}
    for lab in ("RPE1", "HepG2"):
        X = load(d, lab); gR = {g: j for j, g in enumerate(Rk["genes"])}; sg = [g for g in X["genes"] if g in gR]
        ix = np.array([X["genes"].index(g) for g in sg]); ir = np.array([gR[g] for g in sg]); tR = {t: j for j, t in enumerate(Rk["targets"])}
        tot, comp, ref, tr = {}, {}, {}, {}
        for kx, t in enumerate(X["targets"]):
            if t not in tR:
                continue
            vt, vw, rw = units(X, kx, "total"), units(X, kx, "within"), units(Rk, tR[t], "within")
            if vt is None or vw is None or rw is None:
                continue
            tot[t], comp[t], ref[t], tr[t] = vt[:, ix], (vt - vw)[:, ix], rw[:, ir], tier_of(ess.get(t, np.nan))

        def rho(u, r):
            Mx, Mr = float(u[0] @ u[1]), float(r[0] @ r[1])
            return float(u.mean(0) @ r.mean(0)) / np.sqrt(Mx * Mr) if Mx > 0 and Mr > 0 else np.nan
        for nm in ("lt_-1", "-1_-0.5"):
            ts = [t for t in tot if tr[t] == nm]; own, diff = [], []
            for t in ts:
                r0 = rho(tot[t], ref[t]); ra = rho(tot[t] - comp[t], ref[t])
                oth = [u for u in ts if u != t]
                rb = np.nanmedian([rho(tot[t] - comp[oth[i]], ref[t]) for i in rng.integers(0, len(oth), 100)])
                if np.isfinite(r0) and np.isfinite(ra):
                    own.append(ra - r0); diff.append((ra - r0) - (rb - r0))
            out["br"]["%s_%s" % (lab, nm)] = dict(delta_own=boot(lambda v: float(np.median(v)), own),
                                                  own_minus_same_tier=boot(lambda v: float(np.nanmedian(v)), diff))
        print("br", lab, json.dumps(out["br"]), flush=True)

    # bs: denoised benchmark, hESC most-essential tier, common subset, per source
    out["bs"] = {}
    X = load(d, "hESC")
    for src in ("K562", "RPE1", "Jurkat", "HepG2"):
        Sv = load(d, src); gS = {g: j for j, g in enumerate(Sv["genes"])}; sg = [g for g in X["genes"] if g in gS]
        ix = np.array([X["genes"].index(g) for g in sg]); isr = np.array([gS[g] for g in sg]); tS = {t: j for j, t in enumerate(Sv["targets"])}
        rec = {"total": [], "kd": []}
        for kx, t in enumerate(X["targets"]):
            if tier_of(ess.get(t, np.nan)) != "lt_-1" or t not in tS:
                continue
            P_ = units(Sv, tS[t], "total"); Tt = units(X, kx, "total"); Tk = X["kd_r"][kx]
            if P_ is None or Tt is None or not np.isfinite(Tk[0, 0]):
                continue
            PA, PB = P_[0][isr], P_[1][isr]; Q = float(PA @ PB); pm = (PA + PB) / 2
            for key, T in (("total", Tt), ("kd", Tk)):
                A, Bb = T[0][ix], T[1][ix]; Sx = float(A @ Bb); Xc = float(pm @ ((A + Bb) / 2))
                rec[key].append((Sx, Q, Xc))
        o = {}
        for key, v in rec.items():
            v = np.array(v); ok = (v[:, 0] > 0) & (v[:, 1] > 0); v = v[ok]
            o[key] = dict(denoised_scale=boot(lambda s, q: float(np.median(np.sqrt(s / q))), v[:, 0], v[:, 1]),
                          denoised_rel_mse=boot(lambda s, q, x: float(np.median((q - 2 * x + s) / s)), v[:, 0], v[:, 1], v[:, 2]))
        out["bs"][src] = o
        print("bs", src, json.dumps(o), flush=True)
    json.dump(out, open(outf, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k in ("bb", "bc", "bv")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
