"""Fate layer, independent units, hESC (A100; go/no-go check for a fate-layer preprint, 2026-10-18). Z1 (m) reported
that in hESC (Pan et al. 2026 data) essential perturbations are eliminated (~2.8x fewer cells, survivors keep ~59% of
target RNA) and that their response is only ~0.8x that of non-essential ones (inverted U over essentiality), which we
read as "weak responses come from elimination, not resilience". That response ratio used random cell halves; in CD4 the
same ratio rose from ~2x to 8-20x with guide halves (fate_iu_cd4.py), because clonal energy inflates weak perturbations.
Here the same comparison for hESC, from pan_extract_guide.py (key = target * 4 + guide parity * 2 + random half):
  random halves  codes {0, 2} vs {1, 3}      guide halves  codes {0, 1} vs {2, 3}
Effects: mean log1p CP10K per side minus the non-targeting controls of the same codes; genes control mean >= 0.1;
target gene zeroed; referenced to the unexpressed-target mean (control mean of the target gene < 0.01).
Knockdown residual: own-target raw UMIs / (control rate x library), control rate pooled over lanes.
Per DepMap tier (median gene effect over lines: < -1, -1 to -0.5, -0.5 to -0.2, > -0.2): cells, knockdown residual,
median / mean response per design; essential / non-essential ratios. Targets: >= 10 cells on each side of both designs.
Usage: fate_iu_pan.py <pan_agg_guide dir> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
from strata import essentiality, TIERS

DES = {"random": ([0, 2], [1, 3]), "guide": ([0, 1], [2, 3])}


def main():
    d, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
    idx = json.load(open(d + "/index.json")); targets, genes = idx["targets"], idx["genes"]
    T, G = len(targets), len(genes)
    S = np.load(d + "/sums64.npy", mmap_mode="r").reshape(T, 4, G); cnt = np.load(d + "/counts.npy").reshape(T, 4, 6)
    craw, clib = np.load(d + "/ctrl_raw.npy"), np.load(d + "/ctrl_lib.npy")
    ct = targets.index("non-targeting"); n = cnt[:, :, 0]
    cm = np.asarray(S[ct].sum(0)) / n[ct].sum()
    gk = np.where(cm >= 0.1)[0]; kpos = {j: q for q, j in enumerate(gk)}; gpos = {g: j for j, g in enumerate(genes)}
    tids = [t for t in range(T) if t != ct and targets[t] in gpos and all(n[t, c].sum() >= 10 for a, b in DES.values() for c in (a, b))]
    names = [targets[t] for t in tids]
    tx = np.array([cm[gpos[t]] for t in names]); inert, expd = tx < 0.01, tx >= 0.5
    ess = essentiality(); e = np.array([ess.get(t, np.nan) for t in names])
    Sg = np.asarray(S[tids])[:, :, gk].astype(np.float64); nt = n[tids].astype(float)
    Sc = np.asarray(S[ct])[:, gk].astype(np.float64); nc = n[ct].astype(float)
    zc = np.array([kpos.get(gpos[t], -1) for t in names]); rows = np.arange(len(names))
    resp = {}
    for dn, (a, b) in DES.items():
        sides = []
        for codes in (a, b):
            X = Sg[:, codes].sum(1) / nt[:, codes].sum(1)[:, None] - Sc[codes].sum(0) / nc[codes].sum()
            m = zc >= 0; X[rows[m], zc[m]] = 0; sides.append(X)
        A, B = sides; A, B = A - A[inert].mean(0), B - B[inert].mean(0)
        resp[dn] = (A * B).sum(1)
    lam = craw.sum(0) / clib.sum()
    own = cnt[tids][:, :, 5].sum(1); libs = cnt[tids][:, :, 4].sum(1)
    resid = own / np.maximum(np.array([lam[gpos[t]] for t in names]) * libs, 1e-12)
    cells = nt.sum(1)
    res = {"rules": __doc__, "n_targets": len(names), "n_inert": int(inert.sum()), "n_expressed": int(expd.sum()),
           "n_genes": int(len(gk)), "n_with_depmap": int(np.isfinite(e).sum())}
    tiers = {}
    for name, lo, hi in TIERS:
        m = np.isfinite(e) & (e >= lo) & (e < hi) & ~inert; me = m & expd
        tiers[name] = {"n": int(m.sum()), "n_expressed": int(me.sum()), "log10_cells_median": float(np.median(np.log10(cells[m]))),
                       "kd_residual_median_expressed": float(np.median(resid[me])) if me.any() else None,
                       "resp_random_median": float(np.median(resp["random"][m])), "resp_random_mean": float(resp["random"][m].mean()),
                       "resp_guide_median": float(np.median(resp["guide"][m])), "resp_guide_mean": float(resp["guide"][m].mean())}
    tiers["unexpressed_null"] = {"n": int(inert.sum()), "resp_random_median": float(np.median(resp["random"][inert])),
                                 "resp_guide_median": float(np.median(resp["guide"][inert])),
                                 "frac_pos_random": float((resp["random"][inert] > 0).mean()), "frac_pos_guide": float((resp["guide"][inert] > 0).mean())}
    res["tiers"] = tiers
    et, nt_ = tiers["lt_-1"], tiers["gt_-0.2"]
    res["ratio_essential_over_nonessential"] = {k: (et[k] / nt_[k] if nt_[k] > 0 else None) for k in
                                                ("resp_random_median", "resp_guide_median", "resp_random_mean", "resp_guide_mean")}
    print(json.dumps({k: v for k, v in res.items() if k not in ("rules", "tiers")}), flush=True)
    for k, v in tiers.items():
        print("  ", k, json.dumps(v), flush=True)
    json.dump(res, open(out + "/metrics_fate_iu_pan.json", "w"), indent=1)


if __name__ == "__main__":
    main()
