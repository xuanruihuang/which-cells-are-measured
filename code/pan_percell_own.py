"""Per-cell own-target counts in the Pan et al. 2026 hESC screen, for the fate redesign section 3.2 (mixture vs uniform
partial knockdown; notes/fate_analysis_redesign.md, expectations D1-D2 and the thinning control pre-stated, commit 0829f15).
A100. Same cell calls, gene set, guide parity, control rate lambda and candidate targets as pan_extract_kdgroup.py
(candidates: lambda_t x mean control library >= 2). Stores, per cell of a candidate target: target index, guide parity,
own-target raw count c and expected count mu = lambda_t x library; per non-targeting cell: its library and its raw counts
of every candidate target gene (for the control null).
Outputs (out_dir): cells.npz (tc, par, own, mu), ctrl.npz (lib, O = cells x candidates int32, cand = target indices),
index.json. Usage: pan_percell_own.py <raw_dir> <pan_agg_guide_raw dir> <out_dir> <workers>. Exploratory."""
import os, sys, glob, json, time
import numpy as np
from multiprocessing import Pool
from pan_extract import read_lane, setup
from pan_extract_guide import lane_cells2, MIN_MU

G_ = {}


def init(S):
    G_.update(S)


def do_chunk(args):
    files, out, tag = args
    gsel = G_["gsel"]; ct = G_["ctrl_t"]; tcol, lam = G_["tcol"], G_["lam"]
    cand = G_["null_t"]; iscand = np.zeros(len(G_["targets"]), bool); iscand[cand] = True; ccol = tcol[cand]
    acc = {k: [] for k in ("tc", "par", "own", "mu")}; clib, cO = [], []
    t0 = time.time(); ncell = 0
    for fn in files:
        topg, lib, X, bc = lane_cells2(fn, G_["gene_rows"], G_["guide_rows"])
        cells = np.where(topg >= 0)[0]; ncell += len(cells)
        g = topg[cells]; tc = G_["guide_target_idx"][g]; par = G_["guide_par"][g]
        Xs = X[cells][:, gsel].tocsr(); libc = lib[cells]
        m = np.where(iscand[tc])[0]
        if len(m):
            own = np.asarray(Xs[m, tcol[tc[m]]]).ravel()
            acc["tc"].append(tc[m]); acc["par"].append(par[m]); acc["own"].append(own)
            acc["mu"].append(lam[tcol[tc[m]]] * libc[m])
        mc = np.where(tc == ct)[0]
        if len(mc):
            clib.append(libc[mc]); cO.append(Xs[mc][:, ccol].toarray().astype(np.int32))
    np.savez(os.path.join(out, "parts", "cells_%s.npz" % tag),
             **{k: (np.concatenate(v) if v else np.zeros(0)) for k, v in acc.items()})
    np.savez(os.path.join(out, "parts", "ctrl_%s.npz" % tag), lib=np.concatenate(clib) if clib else np.zeros(0),
             O=np.concatenate(cO) if cO else np.zeros((0, len(cand)), np.int32))
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
    S["null_t"] = np.array([k for k in range(len(S["targets"])) if k != ct and S["tcol"][k] >= 0 and lam[S["tcol"][k]] * mean_lib >= 2])
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
    cells = {k: [] for k in ("tc", "par", "own", "mu")}; lib, O = [], []
    for f in sorted(glob.glob(out + "/parts/cells_*.npz")):
        z = np.load(f)
        for k in cells:
            cells[k].append(z[k])
        zc = np.load(f.replace("cells_", "ctrl_")); lib.append(zc["lib"]); O.append(zc["O"])
    np.savez(out + "/cells.npz", **{k: np.concatenate(v) for k, v in cells.items()})
    np.savez(out + "/ctrl.npz", lib=np.concatenate(lib), O=np.concatenate(O), cand=S["null_t"])
    for f in glob.glob(out + "/parts/*.npz"):
        os.remove(f)
    json.dump(dict(targets=S["targets"], lam_cand=[float(lam[S["tcol"][k]]) for k in S["null_t"]],
                   mean_control_library=float(mean_lib)), open(out + "/index.json", "w"))
    print("saved cells", int(sum(len(v) for v in cells["tc"])), "controls", int(sum(len(v) for v in lib)), flush=True)


if __name__ == "__main__":
    main()
