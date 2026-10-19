"""Fate v3r2 external review, check V4 (expectation pre-stated in notes/fate_analysis_redesign.md §7.2, commit f9d9a81):
a selection control whose strength matches the real classification at the observed escape fraction. A100.
Same cells, genes, guide parity, control rate lambda, candidate (null) target genes and classifier as
pan_extract_kdgroup.py. For each non-targeting cell and each candidate target gene, with probability e the cell keeps
its count (an "escaper") and otherwise the count is thinned by Binomial(c, 0.2) (a silenced cell); the result is
classified by the same Poisson rule, and the raw sums of the knocked-down-like subset are accumulated per (e, target
gene, parity). e = 0 reproduces the thinning null (null2) of pan_extract_kdgroup.py with a different random draw.
Outputs: mixnull_raw.npy (len(E) x 2 x null targets x genes), mixnull_cnt.npy (len(E) x 2 x null targets x [n, lib]),
index.json. Usage: pan_extract_mixnull.py <raw_dir> <pan_agg_kdgroup dir> <out_dir> <workers>. Exploratory."""
import os, sys, glob, json, time, zlib
import numpy as np
from multiprocessing import Pool
from scipy.sparse import csr_matrix
from pan_extract import read_lane, setup
from pan_extract_guide import lane_cells2

RKD = 0.2
L9 = np.log(9.0)
EMIX = (0.0, 0.3, 0.5)
G_ = {}


def init(S):
    G_.update(S)


def classify(c, mu):
    llr = -(1 - RKD) * mu + c * np.log(1 / RKD)
    return np.where(llr < -L9, 0, np.where(llr > L9, 1, 2))


def do_chunk(args):
    files, out, tag = args
    gsel = G_["gsel"]; Gs = len(gsel); ct = G_["ctrl_t"]; lam = G_["lam"]; ntcol = G_["tcol"][G_["null_t"]]; Tv = len(ntcol)
    nraw = np.zeros((len(EMIX), 2, Tv, Gs), np.float32); ncnt = np.zeros((len(EMIX), 2, Tv, 2))
    t0 = time.time(); ncell = 0
    for fn in files:
        rng = np.random.default_rng(zlib.crc32(("mix" + os.path.basename(fn)).encode()))
        topg, lib, X, bc = lane_cells2(fn, G_["gene_rows"], G_["guide_rows"])
        cells = np.where(topg >= 0)[0]
        g = topg[cells]; tc = G_["guide_target_idx"][g]; par = G_["guide_par"][g]
        isc = tc == ct
        for p in (0, 1):
            m = cells[isc & (par == p)]
            if not len(m):
                continue
            ncell += len(m)
            Xm = X[m][:, gsel].tocsr(); lm = lib[m]
            O = Xm[:, ntcol].toarray().astype(np.int64); MU = lam[ntcol][None, :] * lm[:, None]
            thin = rng.binomial(O, RKD); u = rng.random(O.shape)
            for i, e in enumerate(EMIX):
                Om = np.where(u < e, O, thin)
                K = csr_matrix(classify(Om, MU) == 0, dtype=np.float32)
                nraw[i, p] += (K.T @ Xm).toarray().astype(np.float32)
                ncnt[i, p, :, 0] += np.asarray(K.sum(0)).ravel(); ncnt[i, p, :, 1] += np.asarray(K.T @ lm).ravel()
    np.save(os.path.join(out, "parts", "nraw_%s.npy" % tag), nraw); np.save(os.path.join(out, "parts", "ncnt_%s.npy" % tag), ncnt)
    return len(files), ncell, time.time() - t0


def main():
    rawdir, kdg, out, nw = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    os.makedirs(out + "/parts", exist_ok=True)
    S = setup(rawdir)
    kidx = json.load(open(kdg + "/index.json"))
    assert S["targets"] == kidx["targets"]
    gpos = {g: k for k, g in enumerate(S["genes"])}
    gsel = [gpos[g] for g in kidx["genes"]]
    S["gsel"] = np.array(gsel); spos = {j: q for q, j in enumerate(gsel)}
    S["tcol"] = np.array([spos.get(gpos[t], -1) if t in gpos else -1 for t in S["targets"]])
    craw, clib = np.load(kdg + "/ctrl_raw.npy"), np.load(kdg + "/ctrl_lib.npy")
    S["lam"] = craw.sum(0) / clib.sum()
    tpos = {t: k for k, t in enumerate(S["targets"])}
    S["null_t"] = np.array([tpos[t] for t in kidx["null_targets"]])
    ft, names, *_ = read_lane(S["files"][0])
    guides = names[S["guide_rows"]]
    par = np.zeros(len(guides), np.int64)
    for t in range(len(S["targets"])):
        ix = np.where(S["guide_target_idx"] == t)[0]
        for r, k in enumerate(ix[np.argsort(guides[ix])]):
            par[k] = r % 2
    S["guide_par"] = par
    print("lanes", len(S["files"]), "genes", len(gsel), "null targets", len(S["null_t"]), "E", EMIX, flush=True)
    chunks = [(S["files"][k::nw], out, "%02d" % k) for k in range(nw) if S["files"][k::nw]]
    with Pool(len(chunks), initializer=init, initargs=(S,)) as pool:
        for nf, nc, dt in pool.imap_unordered(do_chunk, chunks):
            print("chunk", nf, "lanes", nc, "control cells", "%.0fs" % dt, flush=True)
    tot_r = sum(np.load(f).astype(np.float64) for f in sorted(glob.glob(out + "/parts/nraw_*.npy")))
    tot_c = sum(np.load(f) for f in sorted(glob.glob(out + "/parts/ncnt_*.npy")))
    np.save(out + "/mixnull_raw.npy", tot_r.astype(np.float32)); np.save(out + "/mixnull_cnt.npy", tot_c)
    for f in glob.glob(out + "/parts/*.npy"):
        os.remove(f)
    json.dump(dict(emix=list(EMIX), null_targets=kidx["null_targets"], genes=kidx["genes"], rkd=RKD), open(out + "/index.json", "w"))
    print("saved", tot_c[:, :, :, 0].sum(axis=(1, 2)), flush=True)


if __name__ == "__main__":
    main()
