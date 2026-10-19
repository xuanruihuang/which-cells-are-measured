"""Fate v3r2 external review, checks V6, V7, V8, V12 (expectations pre-stated in notes/fate_analysis_redesign.md §7.2,
commit f9d9a81). A100. Fitting rules are those of fate_mixture.py, unchanged (NB, per-gene size from controls, BIC > 10,
>= 30 cells, expected control count >= 3); e is the NB mixture weight of the control-level component, as in Table 2.
V6  ambient / silenced level: Spearman of the fitted silenced level r_kd with DepMap, and the most-essential minus
    non-essential median r_kd with a bootstrap interval; on all fitted targets and on those preferring the mixture.
V7  partial Spearman of e with DepMap, controlling log(mean per-cell expected count) and log(cells); bootstrap intervals
    (2,000) for the median e of the most essential and non-essential strata and their difference; in hESC, Spearman of e
    with the library-normalised abundance within each stratum.
V8  hESC, per guide-parity group: the mixture fitted separately to the cells of each parity (>= 30 each); per-parity
    library-normalised abundance from pan_library_norm.json. Selection predicts higher e for the more depleted parity.
V12 K562 day 6 (essential library) vs day 8 (genome-scale) on shared fitted targets: paired difference in e by stratum.
Usage: fate_rev_mixture.py <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr
import config
import fate_mixture as FM

FI = config.get("FATE_ROOT")
CTX = {"hESC": ("mixture", "hESC", "percell_hesc"), "K562_d8": ("mixture", "K562_gw", "percell_k562gw"),
       "K562_d6": ("mixture_more", "K562ess", "percell_K562ess"), "RPE1": ("mixture_more", "RPE1", "percell_RPE1"),
       "Jurkat": ("mixture_more", "Jurkat", "percell_jurkat"), "HepG2": ("mixture_more", "HepG2", "percell_hepg2"),
       "CD4_rest_D1": ("mixture_more", "CD4D1rest", "percell_CD4D1rest")}
NB = 2000


def boot_med(x, rng):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return dict(n=int(len(x)), median=float(np.median(x)) if len(x) else None, ci95=None)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(NB)]
    return dict(n=int(len(x)), median=float(np.median(x)), ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])


def boot_diff(a, b, rng):
    a = np.asarray(a, float); a = a[np.isfinite(a)]; b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return None
    d = [np.median(a[rng.integers(0, len(a), len(a))]) - np.median(b[rng.integers(0, len(b), len(b))]) for _ in range(NB)]
    return dict(diff=float(np.median(a) - np.median(b)), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                n=[int(len(a)), int(len(b))])


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def partial_spearman(y, x, Z):
    ok = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(Z), 1)
    y, x, Z = rank(y[ok]), rank(x[ok]), np.column_stack([rank(z) for z in Z[ok].T] + [np.ones(ok.sum())])
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]; rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1]), int(ok.sum())


def load_rows():
    m1 = json.load(open(FI + "/mixture/metrics_mixture.json"))["rows"]
    m2 = json.load(open(FI + "/mixture_more/metrics_mixture_more.json"))["rows"]
    return {c: (m1 if f == "mixture" else m2)[lab] for c, (f, lab, _) in CTX.items()}


def mean_mu(pdir):
    idx = json.load(open(pdir + "/index.json")); z = np.load(pdir + "/cells.npz")
    tc, mu = z["tc"], z["mu"]; T = idx["targets"]
    s = np.bincount(tc, weights=mu, minlength=len(T)); n = np.bincount(tc, minlength=len(T))
    return {T[i]: s[i] / n[i] for i in np.where(n > 0)[0]}


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); rng = np.random.default_rng(2026)
    rows = load_rows(); res = {"rules": __doc__, "V6": {}, "V7": {}}
    for c, rr in rows.items():
        mm = mean_mu(FI + "/" + CTX[c][2])
        E = np.array([r["essentiality"] if r["essentiality"] is not None else np.nan for r in rr], float)
        e = np.array([r["nb"]["e"] for r in rr], float); rk = np.array([r["nb"]["r_kd"] for r in rr], float)
        db = np.array([r["nb"]["dbic"] for r in rr], float); n = np.array([r["n"] for r in rr], float)
        mu = np.array([mm.get(r["target"], np.nan) for r in rr], float)
        top, non, mix = E < -1, E > -0.2, db > FM.DBIC
        v6 = {}
        for lab, sel in (("all_fitted", np.ones(len(rr), bool)), ("mixture_preferred", mix)):
            ok = sel & np.isfinite(E) & np.isfinite(rk)
            v6[lab] = dict(spearman_rkd_vs_depmap=float(spearmanr(rk[ok], E[ok])[0]) if ok.sum() > 10 else None, n=int(ok.sum()),
                           median_rkd_most_essential=boot_med(rk[sel & top], rng), median_rkd_non_essential=boot_med(rk[sel & non], rng),
                           diff=boot_diff(rk[sel & top], rk[sel & non], rng))
        res["V6"][c] = v6
        ok = np.isfinite(E) & np.isfinite(e)
        v7 = dict(spearman_e_vs_depmap=float(spearmanr(e[ok], E[ok])[0]) if ok.sum() > 10 else None,
                  partial_spearman_e_vs_depmap=partial_spearman(e, E, np.column_stack([np.log(mu), np.log(n)])),
                  spearman_e_vs_log_mu=float(spearmanr(e[np.isfinite(mu)], np.log(mu[np.isfinite(mu)]))[0]),
                  spearman_e_vs_log_n=float(spearmanr(e, np.log(n))[0]),
                  spearman_depmap_vs_log_mu=float(spearmanr(E[ok & np.isfinite(mu)], np.log(mu[ok & np.isfinite(mu)]))[0]),
                  e_most_essential=boot_med(e[top], rng), e_non_essential=boot_med(e[non], rng), e_diff=boot_diff(e[top], e[non], rng))
        res["V7"][c] = v7
        print("V6", c, json.dumps(v6), flush=True); print("V7", c, json.dumps(v7), flush=True)
    # V7 hESC: e vs library-normalised abundance within strata
    L = {r["target"]: r for r in json.load(open(FI + "/pan_libnorm/pan_library_norm.json"))["rows"]}
    rr = rows["hESC"]; E = np.array([r["essentiality"] if r["essentiality"] is not None else np.nan for r in rr], float)
    e = np.array([r["nb"]["e"] for r in rr]); nm = np.array([L[r["target"]]["norm"] if r["target"] in L and L[r["target"]]["norm"] is not None else np.nan for r in rr], float)
    w = {}
    for lab, lo, hi in FM.TIERS:
        ok = np.isfinite(E) & (E >= lo) & (E < hi) & np.isfinite(nm) & (nm > 0)
        w[lab] = dict(n=int(ok.sum()), spearman_e_vs_abundance=float(spearmanr(e[ok], nm[ok])[0]) if ok.sum() > 10 else None)
    res["V7"]["hESC_within_tier_e_vs_abundance"] = w
    print("V7 hESC within tier", json.dumps(w), flush=True)
    # V8 hESC per parity
    d = FI + "/percell_hesc"; ih = json.load(open(d + "/index.json")); ch = np.load(d + "/ctrl.npz"); z = np.load(d + "/cells.npz")
    T = ih["targets"]; tpos = {t: i for i, t in enumerate(T)}; cand = list(ch["cand"]); lam = np.array(ih["lam_cand"])
    O, lib = ch["O"], ch["lib"]; tc, par, own, mu = z["tc"], z["par"], z["own"].astype(float), z["mu"]
    v8 = []
    for r in rr:
        t = r["target"]; ti = tpos[t]
        if ti not in cand:
            continue
        j = cand.index(ti); k = FM.nb_size(O[:, j].astype(float), lam[j] * lib)
        ep = []
        for p in (0, 1):
            m = (tc == ti) & (par == p)
            ep.append(FM.compare(own[m], mu[m], k) | dict(n=int(m.sum()), residual=float(own[m].sum() / mu[m].sum())) if m.sum() >= FM.MIN_CELLS else None)
        pn = L.get(t, {}).get("parity_norm") or [None, None]
        v8.append(dict(target=t, essentiality=r["essentiality"], e_pooled=r["nb"]["e"], p0=ep[0], p1=ep[1], norm_p0=pn[0], norm_p1=pn[1]))
    res["V8_rows"] = v8
    S = {}
    for lab, lo, hi in FM.TIERS + (("all", -99, 99),):
        de, ok_n = [], 0
        for x in v8:
            Ex = x["essentiality"] if x["essentiality"] is not None else np.nan
            if not (np.isfinite(Ex) and lo <= Ex < hi) or x["p0"] is None or x["p1"] is None:
                continue
            a, b = x["norm_p0"], x["norm_p1"]
            if a is None or b is None or a <= 0 or b <= 0 or a == b:
                continue
            # e of the more depleted parity minus e of the less depleted parity
            lo_p, hi_p = (x["p0"], x["p1"]) if a < b else (x["p1"], x["p0"])
            de.append((lo_p["e"] - hi_p["e"], np.log2(max(a, b) / min(a, b)), lo_p["residual"] - hi_p["residual"]))
        if not de:
            continue
        de = np.array(de)
        S[lab] = dict(n=int(len(de)), frac_more_depleted_has_higher_e=float((de[:, 0] > 0).mean()),
                      median_delta_e=boot_med(de[:, 0], rng), median_log2_abundance_gap=float(np.median(de[:, 1])),
                      median_delta_residual=float(np.median(de[:, 2])),
                      spearman_delta_e_vs_gap=float(spearmanr(de[:, 0], de[:, 1])[0]) if len(de) > 10 else None,
                      frac_more_depleted_has_higher_residual=float((de[:, 2] > 0).mean()))
    # reliability of e between parities (for V3)
    pe = np.array([(x["p0"]["e"], x["p1"]["e"]) for x in v8 if x["p0"] is not None and x["p1"] is not None])
    S["parity_e_pearson"] = float(np.corrcoef(pe[:, 0], pe[:, 1])[0, 1]) if len(pe) > 10 else None; S["n_both_parities"] = int(len(pe))
    res["V8"] = S
    print("V8", json.dumps(S), flush=True)
    # V12 K562 day 6 vs day 8
    a = {r["target"]: r for r in rows["K562_d6"]}; b = {r["target"]: r for r in rows["K562_d8"]}
    sh = sorted(set(a) & set(b)); V = {}
    for lab, lo, hi in FM.TIERS + (("all", -99, 99),):
        ts = [t for t in sh if a[t]["essentiality"] is not None and lo <= a[t]["essentiality"] < hi]
        if not ts:
            continue
        dd = np.array([b[t]["nb"]["e"] - a[t]["nb"]["e"] for t in ts])
        V[lab] = dict(n=len(ts), median_e_d6=float(np.median([a[t]["nb"]["e"] for t in ts])), median_e_d8=float(np.median([b[t]["nb"]["e"] for t in ts])),
                      paired_delta=boot_med(dd, rng), frac_d8_higher=float((dd > 0).mean()),
                      median_cells_d6=float(np.median([a[t]["n"] for t in ts])), median_cells_d8=float(np.median([b[t]["n"] for t in ts])))
    res["V12"] = V
    print("V12", json.dumps(V), flush=True)
    json.dump(res, open(out + "/metrics_rev_mixture.json", "w"), indent=1, default=lambda v: None if v is None else float(v))


if __name__ == "__main__":
    main()
