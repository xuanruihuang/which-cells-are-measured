"""Fate redesign section 3.2: per target, is the distribution of own-target counts across its cells a uniform partial
knockdown or a mixture of knocked-down cells and escapers? (expectations D1-D2 and the thinning control pre-stated in
notes/fate_analysis_redesign.md). A100.
Models for counts c_i with expected control-level counts mu_i:
  uniform:  c_i ~ F(r mu_i)                              (1 parameter)
  mixture:  c_i ~ (1 - e) F(r_kd mu_i) + e F(mu_i)       (2 parameters; escapers at control level)
F = Poisson, or negative binomial with a per-gene size k from the control cells (moments; shared by both models).
Mixture preferred when BIC_uniform - BIC_mixture > 10. Analysable: expected count in an average control cell >= 3 and
>= 30 cells. Control: per target, the same number of control cells, own-gene counts thinned by Binomial(c, 0.5)
(a uniform partial knockdown by construction), fitted the same way.
Usage: fate_mixture.py <percell_hesc dir> <percell_k562gw dir> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from fate_budget import essentiality
import config

MIN_MU, MIN_CELLS, DBIC = 3.0, 30, 10.0
TIERS = (("lt_-1", -9, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, 9))


def lpmf(c, m, k):
    m = np.maximum(m, 1e-12)
    if k is None:
        return c * np.log(m) - m - gammaln(c + 1)
    return gammaln(c + k) - gammaln(k) - gammaln(c + 1) + k * np.log(k / (k + m)) + c * np.log(m / (k + m))


def fit_uniform(c, mu, k):
    if k is None:
        r = max(c.sum() / mu.sum(), 1e-6)
    else:
        r = minimize_scalar(lambda lr: -lpmf(c, np.exp(lr) * mu, k).sum(), bounds=(-12, 1), method="bounded").x
        r = float(np.exp(r))
    return float(lpmf(c, r * mu, k).sum()), r


def fit_mixture(c, mu, k):
    best = (-np.inf, None, None)
    le = lpmf(c, mu, k)
    for e0, r0 in ((0.3, 0.1), (0.7, 0.2), (0.1, 0.5)):
        e, r = e0, r0
        for _ in range(300):
            lk = lpmf(c, r * mu, k)
            a, b = np.log(max(e, 1e-9)) + le, np.log(max(1 - e, 1e-9)) + lk
            mx = np.maximum(a, b); w = np.exp(a - mx) / (np.exp(a - mx) + np.exp(b - mx))
            e_new = float(w.mean()); v = 1 - w
            if k is None:
                r_new = max(float((v * c).sum() / max((v * mu).sum(), 1e-12)), 1e-6)
            else:
                r_new = float(np.exp(minimize_scalar(lambda lr: -(v * lpmf(c, np.exp(lr) * mu, k)).sum(),
                                                     bounds=(-12, 1), method="bounded").x))
            if abs(e_new - e) < 1e-6 and abs(r_new - r) < 1e-6:
                e, r = e_new, r_new; break
            e, r = e_new, r_new
        a, b = np.log(max(e, 1e-12)) + le, np.log(max(1 - e, 1e-12)) + lpmf(c, r * mu, k)
        ll = float(np.logaddexp(a, b).sum())
        if ll > best[0]:
            best = (ll, e, r)
    return best


def compare(c, mu, k):
    n = len(c)
    lu, ru = fit_uniform(c, mu, k); lm, e, r = fit_mixture(c, mu, k)
    lm = max(lm, lu)                                        # the mixture nests the uniform model at e = 0
    dbic = (-2 * lu + np.log(n)) - (-2 * lm + 2 * np.log(n))
    return dict(dbic=float(dbic), r_uniform=ru, e=e, r_kd=r)


def nb_size(O, mu):
    v = ((O - mu) ** 2 - mu).sum(); s = (mu ** 2).sum()
    return float(s / v) if v > 0 else None


def run(ctx, cells, ctrl, lam_cand, cand, targets, ess, rng):
    tc, own, mu = cells["tc"], cells["own"].astype(float), cells["mu"]
    O, lib = ctrl["O"], ctrl["lib"]; mean_lib = lib.mean()
    rows = []
    for j, t in enumerate(cand):
        if lam_cand[j] * mean_lib < MIN_MU:
            continue
        m = tc == t
        if m.sum() < MIN_CELLS:
            continue
        c, u = own[m], mu[m]
        oc = O[:, j].astype(float); umc = lam_cand[j] * lib
        k = nb_size(oc, umc)
        row = dict(target=targets[t], n=int(m.sum()), residual=float(c.sum() / u.sum()), nb_size=k,
                   essentiality=float(ess.get(targets[t], np.nan)))
        row["poisson"] = compare(c, u, None); row["nb"] = compare(c, u, k)
        sel = rng.choice(len(oc), size=min(len(c), len(oc)), replace=False)
        ct = rng.binomial(oc[sel].astype(np.int64), 0.5).astype(float)
        row["ctrl_poisson"] = compare(ct, umc[sel], None); row["ctrl_nb"] = compare(ct, umc[sel], k)
        rows.append(row)
    return rows


def summarize(rows):
    out = {}
    for lab, lo, hi in TIERS + (("all", -99, 99),):
        rr = [r for r in rows if np.isfinite(r["essentiality"]) and lo <= r["essentiality"] < hi] if lab != "all" else rows
        if not rr:
            continue
        d = dict(n=len(rr))
        for key in ("poisson", "nb", "ctrl_poisson", "ctrl_nb"):
            d[key + "_frac_mixture"] = float(np.mean([r[key]["dbic"] > DBIC for r in rr]))
        d["median_residual"] = float(np.median([r["residual"] for r in rr]))
        d["median_e_nb"] = float(np.median([r["nb"]["e"] for r in rr]))
        out[lab] = d
    return out


def main():
    dh, dk, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
    ess = essentiality(config.get("DEPMAP")); rng = np.random.default_rng(57); res = {"rules": __doc__}; allrows = {}
    ih = json.load(open(dh + "/index.json")); ch = np.load(dh + "/ctrl.npz")
    allrows["hESC"] = run("hESC", np.load(dh + "/cells.npz"), ch, np.array(ih["lam_cand"]), ch["cand"], ih["targets"], ess, rng)
    ik = json.load(open(dk + "/index.json")); ck = np.load(dk + "/ctrl.npz")
    allrows["K562_gw"] = run("K562_gw", np.load(dk + "/cells.npz"), ck, ck["lam_cand"], ck["cand"], ik["targets"], ess, rng)
    for ctx, rows in allrows.items():
        res[ctx] = summarize(rows)
        print(ctx, json.dumps(res[ctx]), flush=True)
    res["rows"] = allrows
    json.dump(res, open(out + "/metrics_mixture.json", "w"), indent=1, default=lambda v: None if v is None else float(v))


if __name__ == "__main__":
    main()
