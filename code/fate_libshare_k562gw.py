"""Per-target survivor knockdown residual for the K562 genome-wide screen (Replogle 2022, day 8, raw single cells part 01),
for the library-share analysis of the fate paper (notes/fate_analysis_redesign.md section 3.1). A100.
Light pass: streams the count matrix once and keeps only, per cell, the count of its own target gene and its library size,
plus the summed counts of control cells for the control rate. Residual definition as in pan_fate.py:
  residual = sum of the target's own counts over its cells / (control rate of that gene x sum of their library sizes),
  estimated only when the expected count exceeds 20. Targets absent from the gene list get no residual.
Usage: fate_libshare_k562gw.py <h5ad> <out_dir>."""
import os, sys, json
import numpy as np
import h5py
from screen_io import obs_info, CTRL_NAMES

CHUNK, MIN_EXP = 50000, 20.0


def main():
    path, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
    codes, cats, gem, genes = obs_info(path)
    codes = np.asarray(codes).astype(np.int64)
    with h5py.File(path) as f:
        umi = f["obs"]["UMI_count"][:].astype(np.float64)
    ic = [cats.index(c) for c in CTRL_NAMES if c in cats][0]
    gpos = {g: j for j, g in enumerate(genes)}
    G, K = len(genes), len(cats)
    tg = np.array([gpos.get(c, -1) for c in cats], np.int64)
    own_col = tg[codes]
    ctrl_sum = np.zeros(G); tcount = np.zeros(K)
    with h5py.File(path) as f:
        M = f["layers"]["counts"] if "layers" in f and "counts" in f["layers"] else f["X"]
        dense = isinstance(M, h5py.Dataset)
        N = M.shape[0] if dense else len(M["indptr"]) - 1
        for s in range(0, N, CHUNK):
            e = min(N, s + CHUNK); cc = codes[s:e]; oc = own_col[s:e]
            if dense:
                blk = M[s:e].astype(np.float64)
                ctrl_sum += blk[cc == ic].sum(0)
                own = np.where(oc >= 0, blk[np.arange(e - s), np.maximum(oc, 0)], 0.0)
            else:
                ip = M["indptr"][s:e + 1]
                data = M["data"][ip[0]:ip[-1]].astype(np.float64); ind = M["indices"][ip[0]:ip[-1]].astype(np.int64)
                rr = np.repeat(np.arange(e - s), np.diff(ip))
                mc = cc[rr] == ic
                ctrl_sum += np.bincount(ind[mc], weights=data[mc], minlength=G)
                mt = ind == oc[rr]
                own = np.bincount(rr[mt], weights=data[mt], minlength=e - s)
            tcount += np.bincount(cc, weights=own, minlength=K)
            print("  rows", e, "/", N, flush=True)
    ncell = np.bincount(codes, minlength=K).astype(np.float64)
    tumi = np.bincount(codes, weights=umi, minlength=K)
    rate = ctrl_sum / umi[codes == ic].sum()
    exp = np.where(tg >= 0, tumi * rate[np.maximum(tg, 0)], np.nan)
    kd = np.where(exp > MIN_EXP, tcount / np.where(exp > 0, exp, np.nan), np.nan)
    keep = np.arange(K) != ic
    np.savez(out + "/k562gw_residual.npz", target=np.array(cats)[keep], n=ncell[keep], kd_resid=kd[keep],
             expected=exp[keep], own_count=tcount[keep], in_genes=(tg >= 0)[keep])
    res = {"rules": __doc__, "n_cells": int(N), "n_targets": int(keep.sum()), "ctrl_cells": int(ncell[ic]),
           "n_with_residual": int(np.isfinite(kd[keep]).sum()), "n_in_gene_list": int((tg >= 0)[keep].sum())}
    json.dump(res, open(out + "/metrics_libshare_k562gw.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rules"}), flush=True)


if __name__ == "__main__":
    main()
