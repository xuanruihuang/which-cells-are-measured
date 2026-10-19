"""Per-cell own-target counts with guide-call confidence in the Pan et al. 2026 hESC screen, for the fate redesign §6.2
(B: is baseline escape guide misassignment? expectations B1-B2 pre-stated, commit 0788b9a).
A100. Same cell calls, gene set, guide parity, control rate lambda and candidate targets as pan_extract_kdgroup.py
(candidates: lambda_t x mean control library >= 2). Stores, per cell of a candidate target: target index, guide parity,
own-target raw count c, expected count mu = lambda_t x library, the top guide's UMI count (top) and all guide UMIs (tot);
the guide call itself is the one of pan_extract_guide.lane_cells2 (same MIN_TOP and FRAC).
Outputs (out_dir): cells.npz (tc, par, own, mu, top, tot), index.json.
Usage: pan_percell_guideconf.py <raw_dir> <pan_agg_guide_raw dir> <out_dir> <workers>. Exploratory."""
import os, sys, glob, json, time
import numpy as np
from multiprocessing import Pool
from pan_extract import read_lane, setup, MIN_TOP, FRAC
from pan_extract_guide import MIN_MU
from scipy.sparse import csr_matrix

def lane_cells3(fn, gene_rows, guide_rows):
    """pan_extract_guide.lane_cells2 plus the top guide's UMI and the cell's total guide UMIs."""
    ft, names, shape, ip, ind, dat, bc = read_lane(fn)
    nb = shape[1]
    col = np.repeat(np.arange(nb), np.diff(ip))
    isg = np.zeros(shape[0], bool); isg[gene_rows] = True
    gpos = np.full(shape[0], -1, np.int64); gpos[gene_rows] = np.arange(len(gene_rows))
    gq = np.full(shape[0], -1, np.int64); gq[guide_rows] = np.arange(len(guide_rows))
    mg = isg[ind]
    lib = np.bincount(col[mg], weights=dat[mg], minlength=nb)
    mq = gq[ind] >= 0
    qcol, qid, qv = col[mq], gq[ind[mq]], dat[mq]
    tot = np.bincount(qcol, weights=qv, minlength=nb)
    order = np.lexsort((-qv, qcol)); qcol_s, qid_s, qv_s = qcol[order], qid[order], qv[order]
    first = np.r_[True, qcol_s[1:] != qcol_s[:-1]]
    topv = np.zeros(nb); topg = np.full(nb, -1); topv[qcol_s[first]] = qv_s[first]; topg[qcol_s[first]] = qid_s[first]
    ok = (topv >= MIN_TOP) & (topv >= FRAC * np.maximum(tot, 1)) & (lib > 0)
    topg = np.where(ok, topg, -1)
    X = csr_matrix((dat[mg], (col[mg], gpos[ind[mg]])), shape=(nb, len(gene_rows)))
    return topg, lib, X, bc, topv, tot


G_ = {}


def init(S):
    G_.update(S)


def do_chunk(args):
    files, out, tag = args
    gsel = G_["gsel"]; tcol, lam = G_["tcol"], G_["lam"]
    cand = G_["null_t"]; iscand = np.zeros(len(G_["targets"]), bool); iscand[cand] = True
    acc = {k: [] for k in ("tc", "par", "own", "mu", "top", "tot")}
    t0 = time.time(); ncell = 0
    for fn in files:
        topg, lib, X, bc, topv, tot = lane_cells3(fn, G_["gene_rows"], G_["guide_rows"])
        cells = np.where(topg >= 0)[0]; ncell += len(cells)
        g = topg[cells]; tc = G_["guide_target_idx"][g]; par = G_["guide_par"][g]
        m = np.where(iscand[tc])[0]
        if not len(m):
            continue
        Xs = X[cells[m]][:, gsel].tocsr(); tm = tc[m]
        acc["tc"].append(tm); acc["par"].append(par[m])
        acc["own"].append(np.asarray(Xs[np.arange(len(m)), tcol[tm]]).ravel())
        acc["mu"].append(lam[tcol[tm]] * lib[cells[m]])
        acc["top"].append(topv[cells[m]]); acc["tot"].append(tot[cells[m]])
    np.savez(os.path.join(out, "parts", "cells_%s.npz" % tag),
             **{k: (np.concatenate(v) if v else np.zeros(0)) for k, v in acc.items()})
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
    cells = {k: [] for k in ("tc", "par", "own", "mu", "top", "tot")}
    for f in sorted(glob.glob(out + "/parts/cells_*.npz")):
        z = np.load(f)
        for k in cells:
            cells[k].append(z[k])
    np.savez(out + "/cells.npz", **{k: np.concatenate(v) for k, v in cells.items()})
    for f in glob.glob(out + "/parts/*.npz"):
        os.remove(f)
    json.dump(dict(targets=S["targets"], cand=[int(k) for k in S["null_t"]],
                   lam_cand=[float(lam[S["tcol"][k]]) for k in S["null_t"]], mean_control_library=float(mean_lib)),
              open(out + "/index.json", "w"))
    print("saved cells", int(sum(len(v) for v in cells["tc"])), flush=True)


if __name__ == "__main__":
    main()
