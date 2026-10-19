"""Fate layer, independent units (A100; go/no-go check for a fate-layer preprint, 2026-10-18). Z1 (d) reported that in
primary CD4 T cells essential perturbations are eliminated: ~2.5x fewer cells, survivors keep ~78% of target RNA, and
the response of essential perturbations is only ~2.1x that of non-essential ones (cell lines 4-11x). That response
ratio used cross-half energies of RANDOM cell halves, which in CD4 are dominated by clonal state (~80-90%, preprint):
clonal energy inflates weak and null perturbations and so compresses the essential / non-essential ratio toward 1.
Here, per donor (1, 2) and state (resting, stimulated 48 h), from the per-cell aggregates of cd4_sc_extract_cells.py
(code = guide parity * 4 + r1 * 2 + r2):
  essentiality tier   median DepMap 24Q4 gene effect over lines: < -1, -1 to -0.5, -0.5 to -0.2, > -0.2
  cells               cells per target (all codes)
  knockdown           target gene mean log1p in the target's cells minus control mean (expressed targets: control
                      mean >= 0.5), and the ratio of means (a residual proxy), per guide parity as well
  response            cross-half signal sum_g x_A . x_B of random halves (r1) and of guide halves (parity), referenced
                      to the unexpressed-target mean (null baseline); genes control mean >= 0.1, target gene zeroed,
                      TCR genes excluded
Per tier: median and mean response under each design; essential / non-essential ratios; unexpressed targets as a null
row. Targets: >= 10 cells on each side of both designs.
Usage: fate_iu_cd4.py <out_dir> <label:cells_dir> [<label:cells_dir> ...]. Exploratory."""
import os, sys, json
import numpy as np
from strata import essentiality

DES = {"random": ([0, 1, 4, 5], [2, 3, 6, 7]), "guide": ([0, 1, 2, 3], [4, 5, 6, 7])}
TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf))


def run(d, ess):
    meta = json.load(open(d + "/meta.json")); genes, groups = meta["genes"], meta["groups"]
    S = np.load(d + "/sum8.npy", mmap_mode="r"); n8 = np.load(d + "/n8.npy"); ci = len(groups) - 1
    cm = np.asarray(S[ci].sum(0)) / n8[ci].sum()
    tr = {j for j, g in enumerate(genes) if g.startswith(("TRBC", "TRAC", "TRBV", "TRAV", "TRGC", "TRDC"))}
    gk = np.array([j for j in np.where(cm >= 0.1)[0] if j not in tr]); kpos = {j: q for q, j in enumerate(gk)}
    gpos = {g: j for j, g in enumerate(genes)}
    tids = [i for i, t in enumerate(groups[:-1]) if t in gpos and all(n8[i, c].sum() >= 10 for a, b in DES.values() for c in (a, b))]
    names = [groups[i] for i in tids]
    tx = np.array([cm[gpos[t]] for t in names]); inert, expd = tx < 0.01, tx >= 0.5
    e = np.array([ess.get(t, np.nan) for t in names])
    Sg = np.asarray(S[tids])[:, :, gk].astype(np.float64); nt = n8[tids].astype(float)
    Sc = np.asarray(S[ci])[:, gk].astype(np.float64); nc = n8[ci].astype(float)
    zc = np.array([kpos.get(gpos[t], -1) for t in names]); rows = np.arange(len(names))
    resp = {}
    for dn, (a, b) in DES.items():
        sides = []
        for codes in (a, b):
            X = Sg[:, codes].sum(1) / nt[:, codes].sum(1)[:, None] - Sc[codes].sum(0) / nc[codes].sum()
            m = zc >= 0; X[rows[m], zc[m]] = 0; sides.append(X)
        A, B = sides; A, B = A - A[inert].mean(0), B - B[inert].mean(0)
        resp[dn] = (A * B).sum(1)
    # knockdown of the target gene (all genes of the panel, not only gk)
    tcol = np.array([gpos[t] for t in names])
    tsum = np.array([np.asarray(S[tids[k], :, tcol[k]]) for k in range(len(tids))], dtype=np.float64)   # targets x 8
    ctl = np.asarray(S[ci])[:, tcol].astype(np.float64)                                              # 8 x targets
    def kd(codes):
        mt = tsum[:, codes].sum(1) / nt[:, codes].sum(1); mc = ctl[codes].sum(0) / nc[codes].sum()
        return mt - mc, mt / np.maximum(mc, 1e-9)
    kd_all, res_all = kd(list(range(8))); kd_g0, res_g0 = kd([0, 1, 2, 3]); kd_g1, res_g1 = kd([4, 5, 6, 7])
    cells = nt.sum(1)
    out = {"n_targets": len(names), "n_inert": int(inert.sum()), "n_expressed": int(expd.sum()), "n_genes": int(len(gk)),
           "n_with_depmap": int(np.isfinite(e).sum())}
    tiers = {}
    for name, lo, hi in TIERS:
        m = np.isfinite(e) & (e >= lo) & (e < hi) & ~inert
        me = m & expd
        tiers[name] = {"n": int(m.sum()), "n_expressed": int(me.sum()),
                       "log_cells_median": float(np.median(np.log10(cells[m]))) if m.any() else None,
                       "resp_random_median": float(np.median(resp["random"][m])), "resp_random_mean": float(resp["random"][m].mean()),
                       "resp_guide_median": float(np.median(resp["guide"][m])), "resp_guide_mean": float(resp["guide"][m].mean()),
                       "kd_residual_median_expressed": float(np.median(res_all[me])) if me.any() else None,
                       "kd_residual_guides_median_expressed": [float(np.median(res_g0[me])), float(np.median(res_g1[me]))] if me.any() else None}
    tiers["unexpressed_null"] = {"n": int(inert.sum()), "resp_random_median": float(np.median(resp["random"][inert])),
                                 "resp_random_mean": float(resp["random"][inert].mean()),
                                 "resp_guide_median": float(np.median(resp["guide"][inert])), "resp_guide_mean": float(resp["guide"][inert].mean())}
    out["tiers"] = tiers
    ess_t, non_t = tiers["lt_-1"], tiers["gt_-0.2"]
    out["ratio_essential_over_nonessential"] = {
        "random_median": ess_t["resp_random_median"] / non_t["resp_random_median"],
        "guide_median": ess_t["resp_guide_median"] / non_t["resp_guide_median"] if non_t["resp_guide_median"] > 0 else None,
        "random_mean": ess_t["resp_random_mean"] / non_t["resp_random_mean"],
        "guide_mean": ess_t["resp_guide_mean"] / non_t["resp_guide_mean"] if non_t["resp_guide_mean"] > 0 else None}
    return out


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    ess = essentiality(); res = {"rules": __doc__}
    for spec in sys.argv[2:]:
        lab, d = spec.split(":", 1)
        r = run(d, ess); res[lab] = r
        print(lab, json.dumps({k: v for k, v in r.items() if k != "tiers"}), flush=True)
        for tn, tv in r["tiers"].items():
            print("   ", tn, json.dumps(tv), flush=True)
    json.dump(res, open(out + "/metrics_fate_iu_cd4.json", "w"), indent=1)


if __name__ == "__main__":
    main()
