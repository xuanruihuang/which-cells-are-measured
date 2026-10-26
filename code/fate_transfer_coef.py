"""Single-coefficient transfer model (fate paper v2, Fig. 5a). A source cell type's total effect P predicts the target
cell type's effect T through one coefficient beta per stratum, chosen by least squares over genes and response genes.
With independent replicates on both sides, the denoised least-squares coefficient is beta = sum X / sum Q, where
X = mean(P) . mean(T) and Q = PA . PB (the source's cross-signal, free of its noise), summed over the genes of the
stratum. beta is computed with the target's ground truth from all cells ("total") and from knocked-down cells ("kd"),
on the same genes (those with total, within-cell and knocked-down effects; the benchmark's "common" subset), with a
paired bootstrap over genes (2,000 draws) for beta_total, beta_kd and their ratio. Same data, genes and replicate
definitions as fate_benchmark_denoised.py, whose loading code it uses.
Usage: fate_transfer_coef.py <depmap csv> <vectors dir> <target> <out json> <source> [<source> ...]."""
import sys, json
import numpy as np
from fate_benchmark import essentiality, load, TIERS
from fate_benchmark_denoised import sides


def main():
    dep, d, tlab, outf = sys.argv[1:5]; sources = sys.argv[5:]
    ess = essentiality(dep); X = load(d, tlab); rng = np.random.default_rng(5); res = {"rules": __doc__}
    for src in sources:
        S = load(d, src)
        tS = {t: k for k, t in enumerate(S["targets"])}; gS = {g: j for j, g in enumerate(S["genes"])}
        shared = [g for g in X["genes"] if g in gS]
        ix = np.array([X["genes"].index(g) for g in shared]); isr = np.array([gS[g] for g in shared])
        out = {}
        for nm, lo, hi in TIERS:
            rows = []
            for kx, t in enumerate(X["targets"]):
                e = ess.get(t, np.nan)
                if t not in tS or not np.isfinite(e) or not (lo <= e < hi):
                    continue
                P = sides(S, tS[t], "total")
                Ts = {k: sides(X, kx, k) for k in ("total", "within", "kd")}
                if P is None or any(v is None for v in Ts.values()):
                    continue
                PA, PB = P[0][isr], P[1][isr]; pm = (PA + PB) / 2; Q = float(PA @ PB)
                xt = float(pm @ ((Ts["total"][0][ix] + Ts["total"][1][ix]) / 2))
                xk = float(pm @ ((Ts["kd"][0][ix] + Ts["kd"][1][ix]) / 2))
                rows.append((Q, xt, xk))
            if len(rows) < 10:
                out[nm] = dict(n=len(rows)); continue
            a = np.array(rows); n = len(a)
            def stat(b):
                q = b[:, 0].sum()
                bt, bk = b[:, 1].sum() / q, b[:, 2].sum() / q
                return bt, bk, bk / bt
            bt, bk, r = stat(a)
            boot = np.array([stat(a[rng.integers(0, n, n)]) for _ in range(2000)])
            lo95, hi95 = np.percentile(boot, [2.5, 97.5], axis=0)
            out[nm] = dict(n=n, beta_total=bt, beta_total_ci95=[lo95[0], hi95[0]], beta_kd=bk,
                           beta_kd_ci95=[lo95[1], hi95[1]], ratio_kd_over_total=r, ratio_ci95=[lo95[2], hi95[2]])
            print(src, "->", tlab, nm, json.dumps(out[nm]), flush=True)
        res[src] = out
    json.dump(res, open(outf, "w"), indent=1)


if __name__ == "__main__":
    main()
