"""Fate redesign §6.1 (C): recovery of a known escape fraction by the negative-binomial mixture fit of fate_mixture.py
(expectations C1-C3 pre-stated, commit 0788b9a). A100.
Synthetic targets from control cells of one gene: n cells drawn at random; a fraction e_true keep their counts
(unsilenced), the rest are thinned by Binomial(c, R_TRUE) (silenced). Expected counts mu_i = lambda_g x library_i.
Fitted with fate_mixture.compare (NB size from all control cells of that gene, BIC rule unchanged).
Grid: e_true in E_GRID, n in N_GRID, N_GENES random analysable genes per context (expected count >= 3), one draw each.
Usage: fate_mixture_calib.py <percell_hesc dir> <percell_k562gw dir> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
from fate_mixture import compare, nb_size, MIN_MU, DBIC

E_GRID = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
N_GRID = (50, 150, 400)
N_GENES, R_TRUE = 60, 0.2


def run(ctrl, lam_cand, rng):
    O, lib = ctrl["O"], ctrl["lib"]; mean_lib = lib.mean()
    ok = np.where(np.asarray(lam_cand) * mean_lib >= MIN_MU)[0]
    genes = rng.choice(ok, size=min(N_GENES, len(ok)), replace=False)
    rows = []
    for j in genes:
        oc = O[:, j].astype(np.int64); mu = lam_cand[j] * lib
        k = nb_size(oc.astype(float), mu)
        for n in N_GRID:
            for e in E_GRID:
                sel = rng.choice(len(oc), size=n, replace=False)
                c = oc[sel].copy(); esc = rng.random(n) < e
                c[~esc] = rng.binomial(c[~esc], R_TRUE)
                r = compare(c.astype(float), mu[sel], k)
                rows.append(dict(gene=int(j), n=n, e_true=e, e_hat=r["e"], dbic=r["dbic"], r_kd=r["r_kd"],
                                 mixture=bool(r["dbic"] > DBIC)))
    return rows


def summarize(rows):
    out = {}
    for n in N_GRID:
        for e in E_GRID:
            rr = [r for r in rows if r["n"] == n and r["e_true"] == e]
            eh = np.array([r["e_hat"] for r in rr], float)
            out["n%d_e%.1f" % (n, e)] = dict(m=len(rr), median_bias=float(np.median(eh - e)),
                                             q05=float(np.quantile(eh - e, 0.05)), q95=float(np.quantile(eh - e, 0.95)),
                                             frac_mixture=float(np.mean([r["mixture"] for r in rr])))
    return out


def main():
    dh, dk, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(83); res = {"rules": __doc__}; allrows = {}
    ch = np.load(dh + "/ctrl.npz"); ih = json.load(open(dh + "/index.json"))
    allrows["hESC"] = run(ch, np.array(ih["lam_cand"]), rng)
    ck = np.load(dk + "/ctrl.npz")
    allrows["K562_gw"] = run(ck, ck["lam_cand"], rng)
    for ctx, rows in allrows.items():
        res[ctx] = summarize(rows)
        for key, v in res[ctx].items():
            print(ctx, key, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()}), flush=True)
    res["rows"] = allrows
    json.dump(res, open(out + "/metrics_mixture_calib.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
