"""Per-cell own-target counts in primary CD4+ T cells (Zhu / Marson 2026, one donor-condition h5ad), for the fate
redesign §6.3 (A: mixture across contexts; expectations A1-A4 pre-stated, commit 0788b9a). A100.
Cells, controls and quality filter as cd4_sc_extract_cells.py: low_quality and unassigned cells dropped; controls are
guide_type non-targeting. Two passes over the CSR matrix: (1) control cells' summed counts and libraries give the
control rate lambda per gene; candidates are targets whose gene is measured and whose expected count in an average
control cell is >= 3; (2) per cell of a candidate target its own-target raw count and expected count lambda x library,
and per control cell its library and raw counts of the candidate genes (for the NB size and the thinned control).
Outputs (out_dir): cells.npz (tc, own, mu), ctrl.npz (lib, O, cand, lam_cand), index.json (targets = perturbed genes).
Usage: cd4_percell_own.py <h5ad> <out_dir>. Exploratory."""
import os, sys, json, time
import numpy as np
import h5py
from scipy.sparse import csr_matrix

CHUNK, MIN_MU = 20000, 3.0


def dec(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def cat(o, k):
    return o[k]["codes"][:], dec(o[k]["categories"][:])


def rows(f, r0, r1, G):
    ip = f["X"]["indptr"][r0:r1 + 1]; a, b = int(ip[0]), int(ip[-1])
    X = csr_matrix((f["X"]["data"][a:b].astype(np.float32), f["X"]["indices"][a:b].astype(np.int64), ip - a), shape=(r1 - r0, G))
    X.sum_duplicates()
    return X


def main():
    fn, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True); t0 = time.time()
    with h5py.File(fn, "r") as f:
        o = f["obs"]
        pc, pcats = cat(o, "perturbed_gene_name"); gc, gcats = cat(o, "guide_type")
        lq = o["low_quality"][:].astype(bool)
        genes = dec(f["var"]["gene_name"][:]); N, G = (int(x) for x in f["X"].attrs["shape"])
    ctrl = np.isin(gc, [k for k, x in enumerate(gcats) if "non" in x.lower()]) & ~lq
    tgt = (pc >= 0) & ~ctrl & ~lq
    gpos = {x: k for k, x in enumerate(genes)}
    tg = np.array([gpos.get(x, -1) for x in pcats], np.int64)
    csum = np.zeros(G); clib = []
    with h5py.File(fn, "r") as f:
        for r0 in range(0, N, CHUNK):
            r1 = min(N, r0 + CHUNK); m = np.where(ctrl[r0:r1])[0]
            if not len(m):
                continue
            X = rows(f, r0, r1, G)[m]
            csum += np.asarray(X.sum(0)).ravel(); clib.append(np.asarray(X.sum(1)).ravel())
    clib = np.concatenate(clib); lam = csum / clib.sum(); mean_lib = clib.mean()
    cand = np.array([k for k in range(len(pcats)) if tg[k] >= 0 and lam[tg[k]] * mean_lib >= MIN_MU])
    iscand = np.zeros(len(pcats), bool); iscand[cand] = True; ccol = tg[cand]
    print("cells", N, "controls", int(ctrl.sum()), "candidates", len(cand), "%.0fs" % (time.time() - t0), flush=True)
    acc = {k: [] for k in ("tc", "own", "mu")}; O, L = [], []
    with h5py.File(fn, "r") as f:
        for r0 in range(0, N, CHUNK):
            r1 = min(N, r0 + CHUNK)
            mt = np.where(tgt[r0:r1] & iscand[np.maximum(pc[r0:r1], 0)])[0]; mc = np.where(ctrl[r0:r1])[0]
            if not len(mt) and not len(mc):
                continue
            X = rows(f, r0, r1, G)
            if len(mt):
                Xt = X[mt]; t = pc[r0:r1][mt].astype(np.int64)
                lib = np.asarray(Xt.sum(1)).ravel(); ok = lib > 0
                own = np.asarray(Xt[np.arange(len(mt)), tg[t]]).ravel()
                acc["tc"].append(t[ok]); acc["own"].append(own[ok]); acc["mu"].append(lam[tg[t[ok]]] * lib[ok])
            if len(mc):
                Xc = X[mc]; lib = np.asarray(Xc.sum(1)).ravel()
                O.append(Xc[:, ccol].toarray().astype(np.int32)); L.append(lib)
            if (r0 // CHUNK) % 20 == 0:
                print("  rows", r1, "/", N, "%.0fs" % (time.time() - t0), flush=True)
    np.savez(out + "/cells.npz", **{k: np.concatenate(v) for k, v in acc.items()})
    np.savez(out + "/ctrl.npz", lib=np.concatenate(L), O=np.concatenate(O), cand=cand, lam_cand=lam[ccol])
    json.dump(dict(targets=pcats, mean_control_library=float(mean_lib), file=fn, rules=__doc__), open(out + "/index.json", "w"))
    print("saved", sum(len(v) for v in acc["tc"]), "cells, %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
