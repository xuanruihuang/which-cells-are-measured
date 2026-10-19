"""Per-cell expression of candidate-target cells in the Pan et al. 2026 hESC screen, for the fate redesign section 3.4
(cross-context agreement with a size-matched random-subset null; notes/fate_analysis_redesign.md, commit 7fee77a).
A100. Same cell calls, gene set, guide parity, control rate lambda and candidate targets as pan_extract_kdgroup.py
(candidates here: lambda_t x mean control library >= 3, as fate_kdgroup.py). Stores, per cell of a candidate target:
target index, guide parity, own-target raw count, expected count mu, library, and its raw counts over the gene set of
pan_extract_guide.py (sparse).
Outputs (out_dir): cells.npz (tc, par, own, mu, lib), X.npz (scipy sparse, cells x genes), index.json.
Usage: pan_percell_expr.py <raw_dir> <pan_agg_guide_raw dir> <out_dir> <workers>. Exploratory."""
import os, sys, glob, json, time
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
from pan_extract import read_lane, setup
from pan_extract_guide import lane_cells2, MIN_MU

G_ = {}


def init(S):
    G_.update(S)


def do_chunk(args):
    files, out, tag = args
    gsel = G_["gsel"]; tcol, lam = G_["tcol"], G_["lam"]
    cand = G_["null_t"]; iscand = np.zeros(len(G_["targets"]), bool); iscand[cand] = True
    acc = {k: [] for k in ("tc", "par", "own", "mu", "lib")}; mats = []
    t0 = time.time(); ncell = 0
    for fn in files:
        topg, lib, X, bc = lane_cells2(fn, G_["gene_rows"], G_["guide_rows"])
        cells = np.where(topg >= 0)[0]; ncell += len(cells)
        g = topg[cells]; tc = G_["guide_target_idx"][g]; par = G_["guide_par"][g]
        m = np.where(iscand[tc])[0]
        if not len(m):
            continue
        Xs = X[cells[m]][:, gsel].tocsr()
        tm = tc[m]
        acc["tc"].append(tm); acc["par"].append(par[m]); acc["lib"].append(lib[cells[m]])
        acc["own"].append(np.asarray(Xs[np.arange(len(m)), tcol[tm]]).ravel()); acc["mu"].append(lam[tcol[tm]] * lib[cells[m]])
        mats.append(Xs.astype(np.float32))
    np.savez(os.path.join(out, "parts", "cells_%s.npz" % tag),
             **{k: (np.concatenate(v) if v else np.zeros(0)) for k, v in acc.items()})
    sp.save_npz(os.path.join(out, "parts", "X_%s.npz" % tag), sp.vstack(mats).tocsr() if mats else sp.csr_matrix((0, len(gsel)), dtype=np.float32))
    return len(files), ncell, time.time() - t0


def main():
    rawdir, prev, out, nw = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    os.makedirs(out + "/parts", exist_ok=True)
    S = setup(rawdir)
    pidx = json.load(open(prev + "/index.json"))
    mu = np.load(prev + "/ctrl_mean_prepass.npy")
    gpos = {g: k for k, g in enumerate(S["genes"])}
    gsel = sorted(set(np.where(mu >= MIN_MU)[0].tolist()) | {gpos[t] for t in S["targets"] if t in gpos})
    assert [S["genes"][j] for j in gsel] == pidx["genes"], "gene set differs from pan_extract_guide"
    S["gsel"] = np.array(gsel); spos = {j: q for q, j in enumerate(gsel)}
    S["tcol"] = np.array([spos.get(gpos[t], -1) if t in gpos else -1 for t in S["targets"]])
    lam = np.load(prev + "/ctrl_raw.npy").sum(0) / np.load(prev + "/ctrl_lib.npy").sum(); S["lam"] = lam
    pcnt = np.load(prev + "/counts.npy").reshape(len(S["targets"]), 4, 6)
    ct = S["ctrl_t"]; mean_lib = pcnt[ct, :, 4].sum() / pcnt[ct, :, 0].sum()
    S["null_t"] = np.array([k for k in range(len(S["targets"])) if k != ct and S["tcol"][k] >= 0 and lam[S["tcol"][k]] * mean_lib >= 3])
    ft, names, *_ = read_lane(S["files"][0])
    guides = names[S["guide_rows"]]
    par = np.zeros(len(guides), np.int64)
    for t in range(len(S["targets"])):
        ix = np.where(S["guide_target_idx"] == t)[0]
        for r, k in enumerate(ix[np.argsort(guides[ix])]):
            par[k] = r % 2
    S["guide_par"] = par
    print("lanes", len(S["files"]), "candidates", len(S["null_t"]), "mean control library", round(float(mean_lib)), flush=True)
    chunks = [(S["files"][k::nw], out, "%02d" % k) for k in range(nw) if S["files"][k::nw]]
    with Pool(len(chunks), initializer=init, initargs=(S,)) as pool:
        for nf, nc, dt in pool.imap_unordered(do_chunk, chunks):
            print("chunk", nf, "lanes", nc, "cells", "%.0fs" % dt, flush=True)
    cells = {k: [] for k in ("tc", "par", "own", "mu", "lib")}; mats = []
    for f in sorted(glob.glob(out + "/parts/cells_*.npz")):
        z = np.load(f)
        for k in cells:
            cells[k].append(z[k])
        mats.append(sp.load_npz(f.replace("cells_", "X_")))
    np.savez(out + "/cells.npz", **{k: np.concatenate(v) for k, v in cells.items()})
    sp.save_npz(out + "/X.npz", sp.vstack(mats).tocsr())
    for f in glob.glob(out + "/parts/*.npz"):
        os.remove(f)
    json.dump(dict(targets=S["targets"], genes=pidx["genes"], cand=[int(k) for k in S["null_t"]],
                   mean_control_library=float(mean_lib)), open(out + "/index.json", "w"))
    print("saved cells", int(sum(len(v) for v in cells["tc"])), flush=True)


if __name__ == "__main__":
    main()
