"""Per-cell own-target counts in the K562 genome-wide screen (Replogle 2022, day 8, raw single cells part 01), for the
fate redesign section 3.2 (mixture vs uniform partial knockdown). A100.
Candidates: targets whose gene is in the file and whose expected count in an average control cell is >= 3
(lambda_t x mean control library; lambda from the control cells). Stores, per cell of a candidate target: target index,
own-target raw count and expected count mu = lambda_t x UMI_count; per control cell: UMI_count and the raw counts of all
candidate genes.
Usage: k562gw_percell_own.py <h5ad> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
import h5py
from screen_io import obs_info, CTRL_NAMES

CHUNK, MIN_MU = 50000, 3.0


def main():
    path, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
    codes, cats, gem, genes = obs_info(path)
    codes = np.asarray(codes).astype(np.int64)
    with h5py.File(path) as f:
        umi = f["obs"]["UMI_count"][:].astype(np.float64)
    ic = [cats.index(c) for c in CTRL_NAMES if c in cats][0]
    gpos = {g: j for j, g in enumerate(genes)}
    tg = np.array([gpos.get(c, -1) for c in cats], np.int64)
    crow = np.where(codes == ic)[0]
    with h5py.File(path) as f:
        M = f["X"]; G = M.shape[1]
        ctrl_sum = np.zeros(G)
        for s in range(0, len(crow), CHUNK):
            ctrl_sum += M[np.sort(crow[s:s + CHUNK])].astype(np.float64).sum(0)
    lam = ctrl_sum / umi[crow].sum(); mean_lib = umi[crow].mean()
    cand = np.array([k for k in range(len(cats)) if k != ic and tg[k] >= 0 and lam[tg[k]] * mean_lib >= MIN_MU])
    iscand = np.zeros(len(cats), bool); iscand[cand] = True; ccol = tg[cand]
    print("cells", len(codes), "controls", len(crow), "candidates", len(cand), flush=True)
    acc = {k: [] for k in ("tc", "own", "mu")}; cO, clib = [], []
    with h5py.File(path) as f:
        M = f["X"]; N = M.shape[0]
        for s in range(0, N, CHUNK):
            e = min(N, s + CHUNK); cc = codes[s:e]
            m = np.where(iscand[cc])[0]; mc = np.where(cc == ic)[0]
            if not len(m) and not len(mc):
                continue
            blk = M[s:e]
            if len(m):
                acc["tc"].append(cc[m]); acc["own"].append(blk[m, tg[cc[m]]].astype(np.float64))
                acc["mu"].append(lam[tg[cc[m]]] * umi[s:e][m])
            if len(mc):
                cO.append(blk[mc][:, ccol].astype(np.int32)); clib.append(umi[s:e][mc])
            print("  rows", e, "/", N, flush=True)
    np.savez(out + "/cells.npz", **{k: np.concatenate(v) for k, v in acc.items()})
    np.savez(out + "/ctrl.npz", lib=np.concatenate(clib), O=np.concatenate(cO), cand=cand, lam_cand=lam[ccol])
    json.dump(dict(targets=list(cats), mean_control_library=float(mean_lib), rules=__doc__), open(out + "/index.json", "w"))
    print("saved cells", int(sum(len(v) for v in acc["tc"])), flush=True)


if __name__ == "__main__":
    main()
