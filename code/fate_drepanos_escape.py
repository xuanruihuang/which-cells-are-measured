"""Escape fraction per day within one screen: Drepanos et al. 2026 K562, CRISPRi and CRISPRko, days 4 / 7 / 10 / 14
(5090; readings stated before running). Removes the library confound of the K562
day-6 vs day-8 comparison (two screens, two libraries).
Per day, modality and target: own-target counts c_i of the target's cells; expected control-level counts
mu_i = rate x library size, rate from the cells of the arm-matched reference targets of fate_drepanos.py (same modality,
target itself left out); NB size k from the reference cells (moments). Fit and BIC rule as fate_mixture.py (uniform
partial knockdown vs knocked-down + control-level mixture; mixture preferred when dBIC > 10; >= 30 cells; expected
count in an average reference cell >= 3). Cells above MAX_CELLS are subsampled once (seeded) to bound run time.
Interval: 95% bootstrap over cells (N_BOOT draws). CRISPRko is reported but is not primary: an escaping knockout cell
(in-frame repair) need not keep control-level mRNA, and frameshifted mRNA may or may not decay.
Usage: fate_drepanos_escape.py <drepanos dir> <fate_drepanos metrics json> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fate_drepanos import load, DAYS, MODS

MIN_MU, MIN_CELLS, DBIC, MAX_CELLS, N_BOOT = 3.0, 30, 10.0, 800, 40


def lpmf(c, m, k):
    m = np.maximum(m, 1e-12)
    if k is None:
        return c * np.log(m) - m - gammaln(c + 1)
    return gammaln(c + k) - gammaln(k) - gammaln(c + 1) + k * np.log(k / (k + m)) + c * np.log(m / (k + m))


def fit_uniform(c, mu, k):
    if k is None:
        r = max(c.sum() / mu.sum(), 1e-6)
    else:
        r = float(np.exp(minimize_scalar(lambda lr: -lpmf(c, np.exp(lr) * mu, k).sum(), bounds=(-12, 1), method="bounded").x))
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
    lm = max(lm, lu)
    dbic = (-2 * lu + np.log(n)) - (-2 * lm + 2 * np.log(n))
    return dict(dbic=float(dbic), r_uniform=ru, e=e, r_kd=r)


def nb_size(O, mu):
    v = ((O - mu) ** 2 - mu).sum(); s = (mu ** 2).sum()
    return float(s / v) if v > 0 else None


def main():
    d, prev, outd = sys.argv[1:4]; os.makedirs(outd, exist_ok=True)
    P = json.load(open(prev)); refs = P["reference_targets"]
    rng = np.random.default_rng(7)
    res = {"rules": __doc__, "reference_targets": refs, "days": {}}
    for day in DAYS:
        path = "%s/K562_Day%d.h5" % (d, day)
        import h5py
        with h5py.File(path) as f:
            grps = [g for g in f if g.startswith("counts")]
        res["days"][day] = {}
        for mo in MODS:
            grp = [g for g in grps if g.endswith(mo)] or ["counts"]
            M, genes, guide, mod = load(path, grp[0])
            sel = np.where(mod == mo)[0]
            M, guide = M[sel].tocsc(), guide[sel]
            lib = np.asarray(M.sum(1)).ravel().astype(float)
            tgt = np.array([g.split("_")[0] for g in guide]); gpos = {g: j for j, g in enumerate(genes)}
            out = {}
            for t in sorted(set(tgt)):
                j = gpos.get(t)
                rec = dict(cells=int((tgt == t).sum()))
                others = [r for r in refs[mo] if r != t]
                ri = np.isin(tgt, others)
                if j is None or ri.sum() == 0:
                    rec["skip"] = "no gene or no reference"; out[t] = rec; continue
                col = np.asarray(M[:, j].todense()).ravel().astype(float)
                rate = col[ri].sum() / lib[ri].sum(); rec["ref_mean_expected"] = float(rate * lib[ri].mean())
                if rec["ref_mean_expected"] < MIN_MU:
                    rec["skip"] = "expected count < %g" % MIN_MU; out[t] = rec; continue
                ti = np.where(tgt == t)[0]
                if len(ti) < MIN_CELLS:
                    rec["skip"] = "fewer than %d cells" % MIN_CELLS; out[t] = rec; continue
                if len(ti) > MAX_CELLS:
                    ti = np.sort(rng.choice(ti, MAX_CELLS, replace=False))
                c, mu = col[ti], rate * lib[ti]
                k = nb_size(col[ri], rate * lib[ri])
                rec.update(n_fitted=int(len(ti)), nb_size=k, residual=float(c.sum() / mu.sum()), nb=compare(c, mu, k))
                bs = []
                for _ in range(N_BOOT):
                    b = rng.integers(0, len(c), len(c)); bs.append(compare(c[b], mu[b], k)["e"])
                rec["e_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                out[t] = rec
                print(day, mo, t, rec["cells"], round(rec["residual"], 3), round(rec["nb"]["e"], 3), [round(x, 3) for x in rec["e_ci95"]],
                      round(rec["nb"]["dbic"], 1), flush=True)
            res["days"][day][mo] = out
            del M
    for dd in DAYS:
        for mo in MODS:
            for t, v in res["days"][dd][mo].items():
                v["essentiality"] = P["days"][str(dd)][mo]["targets"].get(t, {}).get("essentiality")
                v["abundance_vs_nonessential"] = P["days"][str(dd)][mo]["targets"].get(t, {}).get("abundance_vs_nonessential")
    json.dump(res, open(os.path.join(outd, "metrics_drepanos_escape.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
