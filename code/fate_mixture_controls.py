"""R4 (fate paper §8; expectations M1-M4 pre-stated in notes/fate_paper_plan.md §8): three controls for the mixture vs
uniform-partial-knockdown test of fate_mixture.py (fitting, BIC > 10 rule and NB size from control cells unchanged).
(a) continuous knockdown: control cells of a gene, each thinned by its own r_i ~ Beta(mean 0.3, sd in SD_GRID), no
    escapers; plus e_true = 0.3 escapers with sd 0.2 heterogeneity. False-mixture rate, e_hat, r_kd_hat.
(b) non-target dispersion baseline (hESC): for each analysable target, up to NG other candidate genes with lambda within
    [0.5, 2] x the target gene's lambda, fitted on the target's cells (mu = lambda_g x library, k from control cells);
    the own-target fit on the same targets alongside.
(c) free escaper level: (1 - e) F(r_kd mu) + e F(r_esc mu), r_esc in [0.37, 1.8]; e and r_esc per target against the
    fixed-level fit, hESC and K562.
Usage: fate_mixture_controls.py <percell_hesc> <percell_expr_hesc> <percell_k562gw> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
import config
from fate_mixture import lpmf, fit_uniform, fit_mixture, compare, nb_size, MIN_MU, MIN_CELLS, DBIC, TIERS
from fate_budget import essentiality

DEPMAP = config.get("DEPMAP")
SD_GRID = (0.0, 0.1, 0.2, 0.3)
N_GRID = (150, 400)
N_GENES, MEAN_R, NG = 40, 0.3, 5


def beta_params(m, s):
    v = s * s; k = m * (1 - m) / v - 1
    return m * k, (1 - m) * k


def synth(ctrl, lam_cand, rng, tag):
    O, lib = ctrl["O"], ctrl["lib"]; mean_lib = lib.mean()
    ok = np.where(np.asarray(lam_cand) * mean_lib >= MIN_MU)[0]
    genes = rng.choice(ok, size=min(N_GENES, len(ok)), replace=False); rows = []
    for j in genes:
        oc = O[:, j].astype(np.int64); mu = np.asarray(lam_cand)[j] * lib; k = nb_size(oc.astype(float), mu)
        for n in N_GRID:
            for sd in SD_GRID:
                for e_true in ((0.0, 0.3) if sd == 0.2 else (0.0,)):
                    sel = rng.choice(len(oc), size=n, replace=False); c = oc[sel].copy()
                    r = np.full(n, MEAN_R) if sd == 0 else np.clip(rng.beta(*beta_params(MEAN_R, sd), size=n), 0.001, 1.0)
                    esc = rng.random(n) < e_true; r[esc] = 1.0
                    c = rng.binomial(c, r)
                    f = compare(c.astype(float), mu[sel], k)
                    rows.append(dict(ctx=tag, gene=int(j), n=n, sd=sd, e_true=e_true, e_hat=f["e"], r_kd=f["r_kd"], dbic=f["dbic"], mixture=bool(f["dbic"] > DBIC)))
    return rows


def summ_synth(rows):
    out = {}
    for ctx in sorted(set(r["ctx"] for r in rows)):
        for n in N_GRID:
            for sd in SD_GRID:
                for e_true in (0.0, 0.3):
                    rr = [r for r in rows if r["ctx"] == ctx and r["n"] == n and r["sd"] == sd and r["e_true"] == e_true]
                    if rr:
                        out["%s|n%d|sd%.1f|e%.1f" % (ctx, n, sd, e_true)] = dict(
                            n_genes=len(rr), frac_mixture=float(np.mean([r["mixture"] for r in rr])),
                            e_hat_median=float(np.median([r["e_hat"] for r in rr])), e_hat_q=[float(np.percentile([r["e_hat"] for r in rr], q)) for q in (10, 90)],
                            r_kd_median=float(np.median([r["r_kd"] for r in rr])))
    return out


def fit_mixture_free(c, mu, k):
    """(1 - e) F(r_kd mu) + e F(r_esc mu), both levels free; EM from three starts; returns (ll, e, r_kd, r_esc)."""
    best = (-np.inf, None, None, None)
    def wfit(w, lo, hi):
        return float(np.exp(minimize_scalar(lambda lr: -(w * lpmf(c, np.exp(lr) * mu, k)).sum(), bounds=(lo, hi), method="bounded").x))
    for e0, r0, s0 in ((0.3, 0.1, 1.0), (0.7, 0.2, 0.8), (0.1, 0.5, 1.2)):
        e, r, s = e0, r0, s0
        for _ in range(300):
            a = np.log(max(e, 1e-9)) + lpmf(c, s * mu, k); b = np.log(max(1 - e, 1e-9)) + lpmf(c, r * mu, k)
            mx = np.maximum(a, b); w = np.exp(a - mx) / (np.exp(a - mx) + np.exp(b - mx))
            e_new = float(w.mean()); r_new = wfit(1 - w, -12, 1); s_new = wfit(w, np.log(0.37), np.log(1.8))
            done = abs(e_new - e) < 1e-6 and abs(r_new - r) < 1e-6 and abs(s_new - s) < 1e-6
            e, r, s = e_new, r_new, s_new
            if done:
                break
        if r > s:                                               # label the lower level as knocked down
            r, s, e = s, r, 1 - e
        a = np.log(max(e, 1e-12)) + lpmf(c, s * mu, k); b = np.log(max(1 - e, 1e-12)) + lpmf(c, r * mu, k)
        ll = float(np.logaddexp(a, b).sum())
        if ll > best[0]:
            best = (ll, e, r, s)
    return best


def free_level(cells, ctrl, lam_cand, cand, targets, ess, tag):
    tc, own, mu = cells["tc"], cells["own"].astype(float), cells["mu"]; O, lib = ctrl["O"], ctrl["lib"]; mean_lib = lib.mean()
    rows = []
    for j, t in enumerate(cand):
        if lam_cand[j] * mean_lib < MIN_MU:
            continue
        m = tc == t
        if m.sum() < MIN_CELLS:
            continue
        c, u = own[m], mu[m]; k = nb_size(O[:, j].astype(float), lam_cand[j] * lib)
        f = compare(c, u, k); ll, e, r, s = fit_mixture_free(c, u, k)
        n = len(c); lu, _ = fit_uniform(c, u, k)
        dbic_free = (-2 * lu + np.log(n)) - (-2 * max(ll, lu) + 3 * np.log(n))
        rows.append(dict(ctx=tag, target=targets[t], n=int(n), essentiality=float(ess.get(targets[t], np.nan)), e_fixed=f["e"], r_kd_fixed=f["r_kd"],
                         dbic_fixed=f["dbic"], e_free=e, r_kd_free=r, r_esc=s, dbic_free=float(dbic_free)))
    return rows


def summ_free(rows):
    out = {}
    for lab, lo, hi in TIERS + (("all", -99, 99),):
        rr = [r for r in rows if np.isfinite(r["essentiality"]) and lo <= r["essentiality"] < hi] if lab != "all" else rows
        if not rr:
            continue
        mix = [r for r in rr if r["dbic_fixed"] > DBIC]
        out[lab] = dict(n=len(rr), n_mixture_fixed=len(mix), e_fixed_median=float(np.median([r["e_fixed"] for r in rr])),
                        e_free_median=float(np.median([r["e_free"] for r in rr])), delta_e_median=float(np.median([r["e_free"] - r["e_fixed"] for r in rr])),
                        r_esc_median=float(np.median([r["r_esc"] for r in rr])), r_esc_q=[float(np.percentile([r["r_esc"] for r in rr], q)) for q in (25, 75)],
                        frac_r_esc_below_0_8=float(np.mean([r["r_esc"] < 0.8 for r in rr])),
                        e_free_median_mixture_targets=float(np.median([r["e_free"] for r in mix])) if mix else None,
                        r_esc_median_mixture_targets=float(np.median([r["r_esc"] for r in mix])) if mix else None,
                        frac_free_preferred_over_fixed=float(np.mean([(r["dbic_free"] - r["dbic_fixed"]) > DBIC for r in rr])))
    return out


def nontarget(pe, ph, ess, rng):
    idx = json.load(open(pe + "/index.json")); genes = idx["genes"]; gpos = {g: j for j, g in enumerate(genes)}
    cells = np.load(pe + "/cells.npz"); tc, own, mu, lib = cells["tc"], cells["own"].astype(float), cells["mu"], cells["lib"]
    X = sp.load_npz(pe + "/X.npz").tocsc()
    ih = json.load(open(ph + "/index.json")); ctrl = np.load(ph + "/ctrl.npz"); cand = ctrl["cand"]; lam = np.array(ih["lam_cand"])
    targets = ih["targets"]; mean_lib = ctrl["lib"].mean(); O, clib = ctrl["O"], ctrl["lib"]
    cand_gene = [targets[t] for t in cand]; cand_pos = {t: j for j, t in enumerate(cand)}
    ksize = {}
    def k_of(j):
        if j not in ksize:
            ksize[j] = nb_size(O[:, j].astype(float), lam[j] * clib)
        return ksize[j]
    rows = []
    for j, t in enumerate(cand):
        if lam[j] * mean_lib < MIN_MU or cand_gene[j] not in gpos:
            continue
        m = np.where(tc == t)[0]
        if len(m) < MIN_CELLS:
            continue
        f_own = compare(own[m], mu[m], k_of(j))
        pool = [q for q in range(len(cand)) if q != j and lam[q] * mean_lib >= MIN_MU and 0.5 * lam[j] <= lam[q] <= 2 * lam[j] and cand_gene[q] in gpos]
        if not pool:
            continue
        pick = rng.choice(pool, size=min(NG, len(pool)), replace=False)
        for q in pick:
            cg = np.asarray(X[m, gpos[cand_gene[q]]].todense()).ravel().astype(float)
            f = compare(cg, lam[q] * lib[m], k_of(q))
            rows.append(dict(target=targets[t], gene=cand_gene[q], n=int(len(m)), essentiality=float(ess.get(targets[t], np.nan)),
                             own_dbic=f_own["dbic"], own_e=f_own["e"], nt_dbic=f["dbic"], nt_e=f["e"], nt_r=f["r_uniform"]))
    return rows


def summ_nt(rows):
    out = {}
    for lab, lo, hi in TIERS + (("all", -99, 99),):
        rr = [r for r in rows if np.isfinite(r["essentiality"]) and lo <= r["essentiality"] < hi] if lab != "all" else rows
        if not rr:
            continue
        tg = {}
        for r in rr:
            tg.setdefault(r["target"], (r["own_dbic"], r["own_e"]))
        out[lab] = dict(n_pairs=len(rr), n_targets=len(tg), nontarget_frac_mixture=float(np.mean([r["nt_dbic"] > DBIC for r in rr])),
                        nontarget_e_median=float(np.median([r["nt_e"] for r in rr])), nontarget_r_median=float(np.median([r["nt_r"] for r in rr])),
                        own_frac_mixture=float(np.mean([v[0] > DBIC for v in tg.values()])), own_e_median=float(np.median([v[1] for v in tg.values()])),
                        frac_targets_any_nontarget_mixture=float(np.mean([any(r["nt_dbic"] > DBIC for r in rr if r["target"] == t) for t in tg])))
    return out


def main():
    ph, pe, pk, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True); ess = essentiality(DEPMAP); rng = np.random.default_rng(11)
    ih = json.load(open(ph + "/index.json")); ch = np.load(ph + "/ctrl.npz"); lam_h = np.array(ih["lam_cand"])
    ik = json.load(open(pk + "/index.json")); ck = np.load(pk + "/ctrl.npz"); lam_k = ck["lam_cand"]
    res = {"rules": __doc__}
    syn = synth(ch, lam_h, rng, "hESC") + synth(ck, lam_k, rng, "K562_gw")
    res["a_continuous"] = summ_synth(syn); print("== (a)", json.dumps(res["a_continuous"], indent=0), flush=True)
    fr = free_level(np.load(ph + "/cells.npz"), ch, lam_h, ch["cand"], ih["targets"], ess, "hESC") + \
         free_level(np.load(pk + "/cells.npz"), ck, lam_k, ck["cand"], ik["targets"], ess, "K562_gw")
    res["c_free_level"] = {ctx: summ_free([r for r in fr if r["ctx"] == ctx]) for ctx in ("hESC", "K562_gw")}
    print("== (c)", json.dumps(res["c_free_level"], indent=0), flush=True)
    nt = nontarget(pe, ph, ess, rng)
    res["b_nontarget_hesc"] = summ_nt(nt); print("== (b)", json.dumps(res["b_nontarget_hesc"], indent=0), flush=True)
    res["rows"] = dict(a=syn, b=nt, c=fr)
    json.dump(res, open(out + "/metrics_mixture_controls.json", "w"), indent=1, default=lambda v: None if v is None else float(v))


if __name__ == "__main__":
    main()
