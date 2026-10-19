"""Fate v3r2 external review, check V11 (expectation pre-stated in notes/fate_analysis_redesign.md §7.2, commit f9d9a81):
composition by pluripotency state in hESC, the analogue of the cell-cycle composition term. A100.
Same cell calls, guide parity and gene set as pan_extract_kdgroup.py. Per cell: state score = mean log1p(CP10K) of
pluripotency markers minus mean log1p(CP10K) of early differentiation markers (markers absent from the data are
dropped). States = tertiles of the score in non-targeting cells (thresholds from the control cells of the first 12
lanes): 0 = lowest pluripotency score, 2 = highest. Outputs: raw sums per (target, parity, state) over the kdgroup gene
set, counts [n, lib] per key, the thresholds and markers used.
Usage: pan_extract_state.py <raw_dir> <pan_agg_kdgroup dir> <out_dir> <workers>. Exploratory."""
import os, sys, glob, json, time
import numpy as np
from multiprocessing import Pool
from scipy.sparse import csr_matrix
from pan_extract import read_lane, setup
from pan_extract_guide import lane_cells2

PLURI = ["POU5F1", "NANOG", "SOX2", "LIN28A", "DPPA4", "TDGF1", "L1TD1", "DNMT3B", "ZFP42", "PRDM14"]
DIFF = ["KRT8", "KRT18", "HAND1", "GATA3", "GATA6", "CDX2", "TBXT", "MIXL1", "PAX6", "SOX17", "VIM", "TFAP2A"]
G_ = {}


def init(S):
    G_.update(S)


def score(X, lib):
    cp = lambda cols: np.log1p(np.asarray(X[:, cols].todense()) / np.maximum(lib, 1)[:, None] * 1e4).mean(1)
    return cp(G_["ip"]) - cp(G_["id"])


def do_chunk(args):
    files, out, tag = args
    T = len(G_["targets"]); gsel = G_["gsel"]; Gs = len(gsel); thr = G_["thr"]
    raw = np.zeros((T * 6, Gs), np.float32); cnt = np.zeros((T * 6, 2))
    t0 = time.time(); ncell = 0
    for fn in files:
        topg, lib, X, bc = lane_cells2(fn, G_["gene_rows"], G_["guide_rows"])
        cells = np.where(topg >= 0)[0]; ncell += len(cells)
        g = topg[cells]; tc = G_["guide_target_idx"][g]; par = G_["guide_par"][g]
        Xc = X[cells]; libc = lib[cells]
        st = np.digitize(score(Xc, libc), thr)
        key = tc * 6 + par * 3 + st
        uk, inv = np.unique(key, return_inverse=True)
        Gm = csr_matrix((np.ones(len(key)), (inv, np.arange(len(key)))), shape=(len(uk), len(key)))
        raw[uk] += (Gm @ Xc[:, gsel]).toarray().astype(np.float32)
        np.add.at(cnt, (key, 0), 1); np.add.at(cnt, (key, 1), libc)
    np.save(os.path.join(out, "parts", "raw_%s.npy" % tag), raw); np.save(os.path.join(out, "parts", "cnt_%s.npy" % tag), cnt)
    return len(files), ncell, time.time() - t0


def main():
    rawdir, kdg, out, nw = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    os.makedirs(out + "/parts", exist_ok=True)
    S = setup(rawdir); kidx = json.load(open(kdg + "/index.json")); assert S["targets"] == kidx["targets"]
    gpos = {g: k for k, g in enumerate(S["genes"])}
    S["gsel"] = np.array([gpos[g] for g in kidx["genes"]])
    pl = [g for g in PLURI if g in gpos]; df = [g for g in DIFF if g in gpos]
    S["ip"] = np.array([gpos[g] for g in pl]); S["id"] = np.array([gpos[g] for g in df])
    ft, names, *_ = read_lane(S["files"][0]); guides = names[S["guide_rows"]]
    par = np.zeros(len(guides), np.int64)
    for t in range(len(S["targets"])):
        ix = np.where(S["guide_target_idx"] == t)[0]
        for r, k in enumerate(ix[np.argsort(guides[ix])]):
            par[k] = r % 2
    S["guide_par"] = par
    G_.update(S); sc = []
    for fn in S["files"][:12]:
        topg, lib, X, bc = lane_cells2(fn, S["gene_rows"], S["guide_rows"])
        m = np.where((topg >= 0))[0]; m = m[S["guide_target_idx"][topg[m]] == S["ctrl_t"]]
        sc.append(score(X[m], lib[m]))
    sc = np.concatenate(sc); S["thr"] = np.percentile(sc, [100 / 3, 200 / 3])
    print("markers", pl, df, "control cells for thresholds", len(sc), "thresholds", S["thr"], flush=True)
    chunks = [(S["files"][k::nw], out, "%02d" % k) for k in range(nw) if S["files"][k::nw]]
    with Pool(len(chunks), initializer=init, initargs=(S,)) as pool:
        for nf, nc, dt in pool.imap_unordered(do_chunk, chunks):
            print("chunk", nf, "lanes", nc, "cells", "%.0fs" % dt, flush=True)
    raw = sum(np.load(f).astype(np.float64) for f in sorted(glob.glob(out + "/parts/raw_*.npy")))
    cnt = sum(np.load(f) for f in sorted(glob.glob(out + "/parts/cnt_*.npy")))
    np.save(out + "/raw_sums.npy", raw.astype(np.float32)); np.save(out + "/counts.npy", cnt)
    for f in glob.glob(out + "/parts/*.npy"):
        os.remove(f)
    json.dump(dict(targets=S["targets"], genes=kidx["genes"], key="target * 6 + parity * 3 + state", counts_cols=["n", "lib"],
                   pluripotency_markers=pl, differentiation_markers=df, thresholds=[float(x) for x in S["thr"]]), open(out + "/index.json", "w"))
    c = cnt.reshape(-1, 2, 3, 2)
    print("saved; cells by state", c[..., 0].sum(axis=(0, 1)), flush=True)


if __name__ == "__main__":
    main()
