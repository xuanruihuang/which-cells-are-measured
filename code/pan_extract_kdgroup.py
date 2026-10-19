"""Pan et al. 2026 hESC re-extracted per (target, guide parity, per-cell knockdown group) (A100; fate paper C6 core;
predictions Q1-Q5 pre-stated in notes/fate_paper_plan.md §3, commit d2d456d).
Same cell calls, genes and guide parity as pan_extract_guide.py (gene set rebuilt from its ctrl_mean_prepass.npy and
checked against its index.json). Control rate lambda per gene from its ctrl_raw / ctrl_lib.
Per cell assigned to target t (own-target raw count c, library L): mu = lambda_t * L;
  log LR = -(1 - RKD) mu + c log(1 / RKD), RKD = 0.2
  group 0 = knocked down (log LR < -log 9), 1 = escaper (> log 9), 2 = undetermined (or target gene not measured).
Non-targeting cells are stored as group 1. Selection-bias null: non-targeting cells classified by the same rule on each
candidate target gene (lambda_t x mean control library >= 2); raw sums of the "knocked-down-like" control subset per
(target gene, parity).
Thinning null (added after the first run, pre-stated in fate_paper_plan.md, commit 291ea33): the same control cells with
the gene's count thinned by Binomial(c, RKD) (simulated knockdown), classified by the same rule; the knocked-down-like
subset is stored as null2; the unthinned escaper-like subset as null3. RNG seeded per lane (crc32 of the file name).
Outputs: raw_sums.npy (targets x 6 x genes; key = target * 6 + parity * 3 + group), counts.npy (targets x 6 x
[n, lib, own, mu]), null_raw.npy (2 x null targets x genes), null_cnt.npy (2 x null targets x [n, lib]),
ctrl_raw.npy / ctrl_lib.npy (per parity), index.json.
Usage: pan_extract_kdgroup.py <raw_dir> <pan_agg_guide_raw dir> <out_dir> <workers>. Exploratory."""
import os, sys, glob, json, time, zlib
import numpy as np
from multiprocessing import Pool
from scipy.sparse import csr_matrix
from pan_extract import read_lane, setup
from pan_extract_guide import lane_cells2, MIN_MU

RKD = 0.2
L9 = np.log(9.0)
G_ = {}


def init(S):
    G_.update(S)


def classify(c, mu):
    llr = -(1 - RKD) * mu + c * np.log(1 / RKD)
    return np.where(llr < -L9, 0, np.where(llr > L9, 1, 2))


def do_chunk(args):
    files, out, tag = args
    T = len(G_["targets"]); gsel = G_["gsel"]; Gs = len(gsel); ct = G_["ctrl_t"]
    tcol, lam = G_["tcol"], G_["lam"]; nt = G_["null_t"]; ntcol = tcol[nt]; Tv = len(nt)
    raw = np.zeros((T * 6, Gs), np.float32); cnt = np.zeros((T * 6, 4))
    nraw = np.zeros((2, Tv, Gs), np.float32); ncnt = np.zeros((2, Tv, 2))
    nraw2 = np.zeros((2, Tv, Gs), np.float32); ncnt2 = np.zeros((2, Tv, 2))
    nraw3 = np.zeros((2, Tv, Gs), np.float32); ncnt3 = np.zeros((2, Tv, 2))
    craw = np.zeros((2, Gs)); clib = np.zeros(2)
    t0 = time.time(); ncell = 0
    for fn in files:
        rng = np.random.default_rng(zlib.crc32(os.path.basename(fn).encode()))
        topg, lib, X, bc = lane_cells2(fn, G_["gene_rows"], G_["guide_rows"])
        cells = np.where(topg >= 0)[0]; ncell += len(cells)
        g = topg[cells]; tc = G_["guide_target_idx"][g]; par = G_["guide_par"][g]
        Xs = X[cells][:, gsel].tocsr(); libc = lib[cells]
        own = np.zeros(len(cells)); mu = np.zeros(len(cells)); has = tcol[tc] >= 0
        if has.any():
            ih = np.where(has)[0]
            own[ih] = np.asarray(Xs[ih, tcol[tc[ih]]]).ravel()
            mu[ih] = lam[tcol[tc[ih]]] * libc[ih]
        grp = np.where(has, classify(own, mu), 2)
        isc = tc == ct; grp[isc] = 1
        key = tc * 6 + par * 3 + grp
        uk, inv = np.unique(key, return_inverse=True)
        Gm = csr_matrix((np.ones(len(key)), (inv, np.arange(len(key)))), shape=(len(uk), len(key)))
        raw[uk] += (Gm @ Xs).toarray().astype(np.float32)
        np.add.at(cnt, (key, 0), 1); np.add.at(cnt, (key, 1), libc); np.add.at(cnt, (key, 2), own); np.add.at(cnt, (key, 3), mu)
        # controls: per parity totals and selection-bias null
        for p in (0, 1):
            m = np.where(isc & (par == p))[0]
            if not len(m):
                continue
            Xm = Xs[m]; lm = libc[m]
            craw[p] += np.asarray(Xm.sum(0)).ravel(); clib[p] += lm.sum()
            O = Xm[:, ntcol].toarray(); MU = lam[ntcol][None, :] * lm[:, None]
            cls = classify(O, MU); cls2 = classify(rng.binomial(O.astype(np.int64), RKD), MU)
            for Kb, nr, nc in ((cls == 0, nraw, ncnt), (cls2 == 0, nraw2, ncnt2), (cls == 1, nraw3, ncnt3)):
                K = csr_matrix(Kb, dtype=np.float32)          # cells x null targets
                nr[p] += (K.T @ Xm).toarray().astype(np.float32)
                nc[p, :, 0] += np.asarray(K.sum(0)).ravel(); nc[p, :, 1] += np.asarray(K.T @ lm).ravel()
    for nm, a in (("raw", raw), ("cnt", cnt), ("nraw", nraw), ("ncnt", ncnt), ("nraw2", nraw2), ("ncnt2", ncnt2),
                  ("nraw3", nraw3), ("ncnt3", ncnt3), ("craw", craw), ("clib", clib)):
        np.save(os.path.join(out, "parts", "%s_%s.npy" % (nm, tag)), a)
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
    assert S["targets"] == pidx["targets"]
    S["gsel"] = np.array(gsel)
    spos = {j: q for q, j in enumerate(gsel)}
    S["tcol"] = np.array([spos.get(gpos[t], -1) if t in gpos else -1 for t in S["targets"]])
    lam = np.load(prev + "/ctrl_raw.npy").sum(0) / np.load(prev + "/ctrl_lib.npy").sum()
    S["lam"] = lam
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
    print("lanes", len(S["files"]), "genes", len(gsel), "targets", len(S["targets"]), "null targets", len(S["null_t"]),
          "mean control library", round(float(mean_lib)), flush=True)
    chunks = [(S["files"][k::nw], out, "%02d" % k) for k in range(nw) if S["files"][k::nw]]
    with Pool(len(chunks), initializer=init, initargs=(S,)) as pool:
        for nf, nc, dt in pool.imap_unordered(do_chunk, chunks):
            print("chunk", nf, "lanes", nc, "cells", "%.0fs" % dt, flush=True)
    T, Gs, Tv = len(S["targets"]), len(gsel), len(S["null_t"])
    tot = {"raw": np.zeros((T * 6, Gs)), "cnt": np.zeros((T * 6, 4)), "nraw": np.zeros((2, Tv, Gs)), "ncnt": np.zeros((2, Tv, 2)),
           "nraw2": np.zeros((2, Tv, Gs)), "ncnt2": np.zeros((2, Tv, 2)), "nraw3": np.zeros((2, Tv, Gs)), "ncnt3": np.zeros((2, Tv, 2)),
           "craw": np.zeros((2, Gs)), "clib": np.zeros(2)}
    for f in sorted(glob.glob(out + "/parts/raw_*.npy")):
        tag = os.path.basename(f)[len("raw_"):]
        for nm in tot:
            tot[nm] += np.load(os.path.join(out, "parts", nm + "_" + tag))
    np.save(out + "/raw_sums.npy", tot["raw"].astype(np.float32)); np.save(out + "/counts.npy", tot["cnt"])
    np.save(out + "/null_raw.npy", tot["nraw"].astype(np.float32)); np.save(out + "/null_cnt.npy", tot["ncnt"])
    np.save(out + "/ctrl_raw.npy", tot["craw"]); np.save(out + "/ctrl_lib.npy", tot["clib"])
    for k in ("2", "3"):
        np.save(out + "/null%s_raw.npy" % k, tot["nraw" + k].astype(np.float32)); np.save(out + "/null%s_cnt.npy" % k, tot["ncnt" + k])
    for f in glob.glob(out + "/parts/*.npy"):
        os.remove(f)
    json.dump(dict(targets=S["targets"], genes=pidx["genes"], null_targets=[S["targets"][k] for k in S["null_t"]],
                   key="target * 6 + guide_parity * 3 + group (0 knocked down, 1 escaper, 2 undetermined)",
                   counts_cols=["n", "lib_sum", "own_target_raw_sum", "expected_own_sum"], rkd=RKD, llr_threshold="log 9",
                   mean_control_library=float(mean_lib)), open(out + "/index.json", "w"))
    c = tot["cnt"].reshape(T, 6, 4)
    print("saved; cells", int(c[:, :, 0].sum()), "by group", [int(c[:, [g, g + 3], 0].sum()) for g in range(3)], flush=True)


if __name__ == "__main__":
    main()
