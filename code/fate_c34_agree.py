"""Fate redesign section 3.4: does removing the selection term (knocked-down cells only) make the hESC response agree
better with K562, beyond what a size-matched random subset of cells does? (expectations G1-G2 and the random-subset
null pre-stated in notes/fate_analysis_redesign.md, commits 0829f15 and 7fee77a). A100.
Genes, effects and the K562 side exactly as fate_kdgroup.py: genes with control >= MIN_CP10K in hESC and both K562
screens; log2 fold change of pseudobulk raw sums vs the non-targeting controls of the same guide parity; target gene zeroed;
K562 vectors from the day-6 and day-8 screens (all cells).
Per target, de-attenuated correlation rho = 0.5 (h0.k8 + h1.k6) / sqrt((h0.h1)(k6.k8)), h0 / h1 the hESC guide-parity
vectors; undefined if a cross-energy is not positive. Versions: all cells; knocked-down cells (log LR rule, r_kd 0.2,
threshold log 9; >= 10 per parity); knocked-down minus the thinning-null vector (sensitivity); 200 random subsets of all
cells, per parity the size of the knocked-down group. Target set fixed: rho defined for all cells and for knocked-down.
Usage: fate_c34_agree.py <percell_expr dir> <pan_agg_guide_raw dir> <pan_agg_kdgroup dir> <samescale2 dir> <out_dir>."""
import os, sys, json
import numpy as np
import scipy.sparse as sp
from strata import essentiality
from fate_samescale import load_k562, lfc, MIN_CP10K

RKD, L9, MIN_KD, NREP = 0.2, np.log(9.0), 10, 200
TIERS = (("lt_-1", -9, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, 9))


def main():
    pe, pdir, kg, kdir, out = sys.argv[1:6]; os.makedirs(out, exist_ok=True)
    idx = json.load(open(pe + "/index.json")); pt, pg = idx["targets"], idx["genes"]; T = len(pt)
    cells = np.load(pe + "/cells.npz"); X = sp.load_npz(pe + "/X.npz").tocsr()
    tc, par, own, mu, lib = (cells[k] for k in ("tc", "par", "own", "mu", "lib"))
    llr = -(1 - RKD) * mu + own * np.log(1 / RKD); kd = llr < -L9
    # gene filter identical to fate_kdgroup.py
    pidx = json.load(open(pdir + "/index.json")); assert pidx["genes"] == pg
    praw = np.load(pdir + "/raw_sums.npy", mmap_mode="r").reshape(T, 4, len(pg)); pcnt = np.load(pdir + "/counts.npy").reshape(T, 4, 6)
    K6, K8 = load_k562(kdir + "/K6.npz"), load_k562(kdir + "/K8.npz")
    gset = sorted(set(K6["genes"]) & set(K8["genes"])); pgi = {g: j for j, g in enumerate(pg)}
    k6i, k8i = {g: j for j, g in enumerate(K6["genes"])}, {g: j for j, g in enumerate(K8["genes"])}
    ct = pt.index("non-targeting")
    hc = np.array([praw[ct, :, pgi[g]].sum() for g in gset]) / pcnt[ct, :, 4].sum() * 1e4
    c6 = np.array([K6["raw"][-1, :, k6i[g]].sum() for g in gset]) / K6["lib"][-1].sum() * 1e4
    c8 = np.array([K8["raw"][-1, :, k8i[g]].sum() for g in gset]) / K8["lib"][-1].sum() * 1e4
    keep = np.where((hc >= MIN_CP10K) & (c6 >= MIN_CP10K) & (c8 >= MIN_CP10K))[0]
    genes = [gset[j] for j in keep]; gq = {g: q for q, g in enumerate(genes)}
    hcol, c6col, c8col = (np.array([d[g] for g in genes]) for d in (pgi, k6i, k8i))
    kgi = json.load(open(kg + "/index.json")); assert kgi["targets"] == pt
    craw, clib = np.load(kg + "/ctrl_raw.npy"), np.load(kg + "/ctrl_lib.npy")
    N2R, N2C = np.load(kg + "/null2_raw.npy", mmap_mode="r"), np.load(kg + "/null2_cnt.npy")
    nidx = {t: k for k, t in enumerate(kgi["null_targets"])}
    shared = set(K6["targets"][:-1]) & set(K8["targets"][:-1])
    ess = essentiality(); rng = np.random.default_rng(71)
    Xh = X[:, hcol].tocsr()
    order = np.argsort(tc, kind="stable"); bounds = np.searchsorted(tc[order], np.arange(T + 1))

    def zero(v, t):
        q = gq.get(t)
        if q is not None:
            v = v.copy(); v[q] = 0
        return v

    def hv(rows, p, t):
        return zero(lfc(np.asarray(Xh[rows].sum(0)).ravel(), lib[rows].sum(), craw[p, hcol], clib[p]), t)

    def rho(h0, h1, k6, k8):
        a, b = h0 @ h1, k6 @ k8
        return float(0.5 * (h0 @ k8 + h1 @ k6) / np.sqrt(a * b)) if a > 0 and b > 0 else np.nan

    rows = []
    for t in sorted(shared):
        if t not in pgi:
            continue
        ti = pt.index(t)
        r_ = order[bounds[ti]:bounds[ti + 1]]
        if not len(r_):
            continue
        P = [r_[par[r_] == p] for p in (0, 1)]; Kd = [x[kd[x]] for x in P]
        if min(len(x) for x in Kd) < MIN_KD:
            continue
        k6v = zero(lfc(K6["raw"][K6["targets"].index(t)][:, c6col].sum(0), K6["lib"][K6["targets"].index(t)].sum(),
                       K6["raw"][-1][:, c6col].sum(0), K6["lib"][-1].sum()), t)
        k8v = zero(lfc(K8["raw"][K8["targets"].index(t)][:, c8col].sum(0), K8["lib"][K8["targets"].index(t)].sum(),
                       K8["raw"][-1][:, c8col].sum(0), K8["lib"][-1].sum()), t)
        ha = [hv(P[p], p, t) for p in (0, 1)]; hk = [hv(Kd[p], p, t) for p in (0, 1)]
        r_all, r_kd = rho(ha[0], ha[1], k6v, k8v), rho(hk[0], hk[1], k6v, k8v)
        r_kdn = np.nan
        if t in nidx and min(N2C[p, nidx[t], 0] for p in (0, 1)) >= MIN_KD:
            nv = [zero(lfc(np.asarray(N2R[p, nidx[t]])[hcol], N2C[p, nidx[t], 1], craw[p, hcol], clib[p]), t) for p in (0, 1)]
            r_kdn = rho(hk[0] - nv[0], hk[1] - nv[1], k6v, k8v)
        rr = np.full(NREP, np.nan)
        if np.isfinite(r_all) and np.isfinite(r_kd):
            for b in range(NREP):
                hs = [hv(rng.choice(P[p], size=len(Kd[p]), replace=False), p, t) for p in (0, 1)]
                rr[b] = rho(hs[0], hs[1], k6v, k8v)
        rows.append(dict(target=t, essentiality=float(ess.get(t, np.nan)), n_all=int(len(r_)), n_kd=int(sum(len(x) for x in Kd)),
                         rho_all=r_all, rho_kd=r_kd, rho_kd_null2=r_kdn, rho_rand=rr))
        print(t, len(rows), round(r_all, 3) if np.isfinite(r_all) else None, round(r_kd, 3) if np.isfinite(r_kd) else None, flush=True)
    res = {"rules": __doc__, "n_genes": len(genes), "n_rows": len(rows)}
    E = np.array([r["essentiality"] for r in rows]); RA = np.array([r["rho_all"] for r in rows])
    RK = np.array([r["rho_kd"] for r in rows]); RN = np.array([r["rho_kd_null2"] for r in rows])
    RR = np.array([r["rho_rand"] for r in rows])
    for lab, lo, hi in TIERS + (("all", -99, 99),):
        m = np.isfinite(RA) & np.isfinite(RK) & ((E >= lo) & (E < hi) if lab != "all" else True)
        if m.sum() < 5:
            res[lab] = dict(n=int(m.sum())); continue
        d_obs = float(np.median(RK[m]) - np.median(RA[m]))
        null = np.nanmedian(RR[m], axis=0) - np.median(RA[m])
        mn = m & np.isfinite(RN)
        res[lab] = dict(n=int(m.sum()), median_rho_all=float(np.median(RA[m])), median_rho_kd=float(np.median(RK[m])),
                        delta=d_obs, null_median=float(np.nanmedian(null)), null_q975=float(np.nanpercentile(null, 97.5)),
                        frac_null_ge_obs=float(np.mean(null >= d_obs)),
                        delta_null2_corrected=float(np.median(RN[mn]) - np.median(RA[mn])) if mn.sum() >= 5 else None,
                        n_null2=int(mn.sum()))
        print(lab, json.dumps(res[lab]), flush=True)
    json.dump(res, open(out + "/metrics_c34_agree.json", "w"), indent=1)
    np.savez(out + "/c34_rows.npz", target=np.array([r["target"] for r in rows]), essentiality=E, rho_all=RA, rho_kd=RK,
             rho_kd_null2=RN, rho_rand=RR)


if __name__ == "__main__":
    main()
