"""Escape fraction per perturbation from per-cell counts of the targeted gene (Box 1, recommendation 1).

For each perturbation, the counts c_i of its own target gene in its cells, with control-level expectations mu_i
(control rate of that gene x the cell's library size), are fitted by two models:
  uniform:  c_i ~ NB(r mu_i, k)                                  every cell partly silenced
  mixture:  c_i ~ (1 - e) NB(r_kd mu_i, k) + e NB(mu_i, k)       a fraction e of cells at control level
The size k is estimated for that gene from control cells by moments (Var = mu + mu^2 / k) and shared by both models;
the mixture is fitted by expectation-maximisation from three starting points. The mixture is preferred when
BIC_uniform - BIC_mixture > 10. A thinned-control null is fitted the same way for every perturbation: as many control
cells as the perturbation has, their counts thinned by Binomial(c, 0.5), a uniform partial knockdown by construction;
it should not prefer the mixture.

The fitting code is copied unchanged from code/fate_mixture.py, which produced the paper's numbers
(checks/check_escape_fraction.py verifies that both give the same e and BIC difference on the paper's data).

Output, one row per perturbation:
  e                 fraction of measured cells at control level (unsilenced: escaped, misassigned or not silenced)
  dbic              BIC_uniform - BIC_mixture; mixture_detected = dbic > 10
  dilution          (1 - e)^2, the factor by which escape scales a response measured on all cells relative to
                    silenced cells; log2_dilution = log2 of it
  r_kd              residual expression of silenced cells (fraction of control level)
  residual          sum c / sum mu over all cells (the mean survivor residual)
  reliable          n >= 150: e unbiased to about +-0.08 (90% interval) in the paper's recovery test; at n = 50 about
                    +-0.15, and detection falls at e near 0.9
  null_dbic         thinned-control BIC difference (null_mixture = null_dbic > 10 flags a gene the NB fits poorly)
  e_purity_low/mid/high   with a purity column: e in each third of the cells ranked by guide purity (fitted where a
                    third has >= 30 cells); restricting to unambiguous guide calls removes most of the baseline e
                    but not the part that rises with the cost of the perturbation
Perturbations are fitted when they have >= 30 cells and the gene's mean control expectation is >= 3; others are listed
with the reason in `note`.

Usage:
  escape_fraction.py <cells.csv[.gz]> <controls.csv[.gz]> <out.tsv> [--seed N] [--min-purity P] [--poisson]
cells:     perturbation, count, and either expected or library; optional gene (default: the perturbation name, for a
           perturbation that targets one gene) and purity (assigned guide's UMIs / all guide UMIs of the cell)
controls:  gene, count, and either expected or library; one row per control cell and gene, zero counts included.
           With `library` (in both files) the control rate of a gene is sum(count) / sum(library) over its control
           rows and expected = rate x library.
Requires numpy and scipy."""
import argparse
import csv
import gzip
import sys
from collections import OrderedDict

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

MIN_MU, MIN_CELLS, DBIC = 3.0, 30, 10.0
RELIABLE_CELLS = 150


# ---- fitting: copied unchanged from code/fate_mixture.py ----
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
# ---- end of the copied code ----


def purity_thirds(purity, rng):
    """Thirds of the cells by purity (ties broken at random), as in code/fate_mixture_guideconf.py."""
    rk = np.argsort(np.argsort(purity + 1e-9 * rng.random(len(purity))))
    return (3 * rk) // len(purity)


def fit_perturbation(c, mu, ctrl_c, ctrl_mu, rng, purity=None, poisson=False, tiebreak_rng=None):
    """All quantities for one perturbation. c, mu: its cells; ctrl_c, ctrl_mu: control cells of the same gene."""
    c, mu = np.asarray(c, float), np.asarray(mu, float)
    ctrl_c, ctrl_mu = np.asarray(ctrl_c, float), np.asarray(ctrl_mu, float)
    n = len(c); mean_ctrl = float(ctrl_mu.mean()) if len(ctrl_mu) else np.nan
    row = OrderedDict(n_cells=n, mean_control_expected=mean_ctrl,
                      residual=float(c.sum() / mu.sum()) if mu.sum() > 0 else np.nan)
    notes = []
    if not len(ctrl_c):
        notes.append("no control cells for this gene")
    elif mean_ctrl < MIN_MU:
        notes.append("mean control expectation < %g: too few counts per cell to separate the groups" % MIN_MU)
    if n < MIN_CELLS:
        notes.append("fewer than %d cells" % MIN_CELLS)
    k = None if poisson or not len(ctrl_c) else nb_size(ctrl_c, ctrl_mu)
    row["nb_size"] = np.nan if k is None else k
    if notes:
        row["note"] = "; ".join(notes)
        return row
    if k is None and not poisson:
        notes.append("control counts not overdispersed: Poisson used")
    f = compare(c, mu, k)
    e = float(f["e"])
    row.update(e=e, dbic=f["dbic"], mixture_detected=f["dbic"] > DBIC, dilution=(1 - e) ** 2,
               log2_dilution=float(np.log2(max((1 - e) ** 2, 1e-12))), r_kd=f["r_kd"], r_uniform=f["r_uniform"],
               reliable=n >= RELIABLE_CELLS)
    sel = rng.choice(len(ctrl_c), size=min(n, len(ctrl_c)), replace=False)
    ct = rng.binomial(ctrl_c[sel].astype(np.int64), 0.5).astype(float)
    nul = compare(ct, ctrl_mu[sel], k)
    row.update(null_dbic=nul["dbic"], null_mixture=nul["dbic"] > DBIC)
    if n < RELIABLE_CELLS:
        notes.append("fewer than %d cells: e imprecise" % RELIABLE_CELLS)
    if e > 0.8:
        notes.append("e > 0.8: few silenced cells, detection unreliable")
    if row["null_mixture"]:
        notes.append("thinned control prefers the mixture: NB fits this gene poorly")
    if purity is not None:
        third = purity_thirds(np.asarray(purity, float), tiebreak_rng or rng)
        for q, lab in enumerate(("low", "mid", "high")):
            m = third == q
            row["n_purity_" + lab] = int(m.sum())
            row["e_purity_" + lab] = compare(c[m], mu[m], k)["e"] if m.sum() >= MIN_CELLS else np.nan
    row["note"] = "; ".join(notes)
    return row


def read_table(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as f:
        sample = f.read(4096); f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        rd = csv.DictReader(f, dialect=dialect)
        cols = {k: [] for k in rd.fieldnames}
        for r in rd:
            for k in cols:
                cols[k].append(r[k])
    return cols


def groups(key, *labels):
    """Row indices per distinct key, in order of first appearance; keyed by the key or by the labels' values."""
    u, first, inv = np.unique(key, return_index=True, return_inverse=True)
    order = np.argsort(inv, kind="stable"); bounds = np.searchsorted(inv[order], np.arange(len(u) + 1))
    out = OrderedDict()
    for j in np.argsort(first):
        idx = order[bounds[j]:bounds[j + 1]]
        out[tuple(l[idx[0]] for l in labels) if labels else u[j]] = idx
    return out


def run(cells_path, ctrl_path, seed=57, min_purity=None, poisson=False):
    C, K = read_table(cells_path), read_table(ctrl_path)
    for need, tab, name in (("perturbation", C, "cells"), ("count", C, "cells"), ("gene", K, "controls"),
                            ("count", K, "controls")):
        if need not in tab:
            raise SystemExit("%s file has no column '%s'" % (name, need))
    pert = np.array(C["perturbation"]); gene = np.array(C["gene"]) if "gene" in C else pert
    cnt = np.array(C["count"], float)
    kg, kc = np.array(K["gene"]), np.array(K["count"], float)
    if "expected" in C and "expected" in K:
        mu, kmu = np.array(C["expected"], float), np.array(K["expected"], float)
    elif "library" in C and "library" in K:
        lib, klib = np.array(C["library"], float), np.array(K["library"], float)
        ug, inv = np.unique(kg, return_inverse=True)
        rate = dict(zip(ug, np.bincount(inv, kc) / np.bincount(inv, klib)))
        mu = np.array([rate.get(g, np.nan) for g in gene]) * lib
        kmu = np.array([rate[g] for g in kg]) * klib
    else:
        raise SystemExit("give 'expected' in both files, or 'library' in both files")
    pur = np.array(C["purity"], float) if "purity" in C else None
    if pur is not None and min_purity is not None:
        keep = pur >= min_purity
        pert, gene, cnt, mu, pur = pert[keep], gene[keep], cnt[keep], mu[keep], pur[keep]
    rng = np.random.default_rng(seed)
    ctrl_idx = groups(kg)
    rows = []
    for (p, g), m in groups(np.char.add(np.char.add(pert.astype(str), "\x00"), gene.astype(str)), pert, gene).items():
        ci = ctrl_idx.get(g, np.zeros(0, int))
        r = fit_perturbation(cnt[m], mu[m], kc[ci], kmu[ci], rng, purity=None if pur is None else pur[m],
                             poisson=poisson)
        rows.append(OrderedDict([("perturbation", p), ("gene", g)] + list(r.items())))
    return rows


COLUMNS = ("perturbation", "gene", "n_cells", "e", "dbic", "mixture_detected", "dilution", "log2_dilution", "reliable",
           "r_kd", "residual", "r_uniform", "nb_size", "mean_control_expected", "null_dbic", "null_mixture",
           "n_purity_low", "e_purity_low", "n_purity_mid", "e_purity_mid", "n_purity_high", "e_purity_high", "note")


def write(rows, path):
    cols = [c for c in COLUMNS if any(c in r for r in rows)]

    def fmt(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "NA"
        if isinstance(v, (bool, np.bool_)):
            return "TRUE" if v else "FALSE"
        if isinstance(v, float):
            return "%.4g" % v
        return str(v)
    with open(path, "w", newline="") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(fmt(r.get(c)) for c in cols) + "\n")


def summary(rows):
    fit = [r for r in rows if "e" in r]
    out = ["perturbations %d, fitted %d" % (len(rows), len(fit))]
    if fit:
        e = np.array([r["e"] for r in fit])
        out.append("mixture detected %d / %d; thinned-control null %d / %d" % (
            sum(r["mixture_detected"] for r in fit), len(fit), sum(r["null_mixture"] for r in fit), len(fit)))
        out.append("e median %.3f (quartiles %.3f-%.3f); reliable (>= %d cells) %d" % (
            np.median(e), np.quantile(e, 0.25), np.quantile(e, 0.75), RELIABLE_CELLS, sum(r["reliable"] for r in fit)))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cells"); ap.add_argument("controls"); ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=57, help="random seed of the thinned-control null (default 57)")
    ap.add_argument("--min-purity", type=float, default=None,
                    help="keep only cells whose guide purity is at least this value before fitting")
    ap.add_argument("--poisson", action="store_true", help="Poisson instead of NB (not recommended; for comparison)")
    a = ap.parse_args(argv)
    rows = run(a.cells, a.controls, seed=a.seed, min_purity=a.min_purity, poisson=a.poisson)
    write(rows, a.out)
    print(summary(rows))
    print("written", a.out)


if __name__ == "__main__":
    main()
