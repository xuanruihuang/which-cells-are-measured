"""Pan et al. 2026 hESC CRISPRi Perturb-seq (GSE295214) re-extracted per (target, guide parity, random half), so that
guide halves (independent transductions; most targets have two guide elements) can be compared with random cell halves
on the same cells. Same cell calls, normalization and phase calls as pan_extract.py (imported); guide parity = rank of
the called guide's name among the target's guides, mod 2 (non-targeting guides likewise). Accumulated genes: control
prepass mean log1p >= MIN_MU (to fit memory), plus every target gene present.
Outputs: sums (targets x 4 x genes, key = target * 4 + parity * 2 + half), counts (targets x 4 x [n, G1, S, G2M,
lib_sum, own_target_raw_sum]), ctrl_raw / ctrl_lib per lane, index.json.
Usage: pan_extract_guide.py <raw_dir> <out_dir> <workers>. Exploratory."""
import os, sys, glob, json, zlib, time
import numpy as np
from multiprocessing import Pool
from scipy.sparse import csr_matrix
from pan_extract import read_lane, setup, phase_sets, MIN_TOP, FRAC

MIN_MU = 0.05
G_ = {}


def lane_cells2(fn, gene_rows, guide_rows):
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
    return topg, lib, X, bc


def init(S, P):
    G_.update(S); G_.update(P)


def do_chunk(args):
    files, out, tag = args
    T, NL = len(G_["targets"]), len(G_["files"]); gsel = G_["gsel"]; Gs = len(gsel)
    acc = np.zeros((T * 4, Gs), np.float32); cnt = np.zeros((T * 4, 6)); craw = np.zeros((NL, Gs)); clib = np.zeros(NL)
    raw = np.zeros((T * 4, Gs), np.float32) if G_.get("with_raw") else None
    lane_of = {f: k for k, f in enumerate(G_["files"])}
    tgpos = {t: k for k, t in enumerate(G_["genes"])}
    tcol = np.array([tgpos.get(t, -1) for t in G_["targets"]])
    t0 = time.time(); ncell = 0
    for fn in files:
        L = lane_of[fn]
        topg, lib, X, bc = lane_cells2(fn, G_["gene_rows"], G_["guide_rows"])
        cells = np.where(topg >= 0)[0]; ncell += len(cells)
        g = topg[cells]; tc = G_["guide_target_idx"][g]; par = G_["guide_par"][g]
        Xc = X[cells]; libc = lib[cells]
        Y = Xc.multiply(1e4 / libc[:, None]).tocsr(); Y.data = np.log1p(Y.data)
        sS = np.asarray(Y[:, G_["iS"]].mean(1)).ravel() - np.asarray(Y[:, G_["cS"]].mean(1)).ravel()
        sG = np.asarray(Y[:, G_["iG"]].mean(1)).ravel() - np.asarray(Y[:, G_["cG"]].mean(1)).ravel()
        ph = np.where((sS <= 0) & (sG <= 0), 0, np.where(sS >= sG, 1, 2))
        lane_name = os.path.basename(fn)
        half = np.array([zlib.crc32((lane_name + b).encode()) & 1 for b in bc[cells]])
        key = tc * 4 + par * 2 + half
        uk, inv = np.unique(key, return_inverse=True)
        Gm = csr_matrix((np.ones(len(key)), (inv, np.arange(len(key)))), shape=(len(uk), len(key)))
        acc[uk] += (Gm @ Y[:, gsel]).toarray().astype(np.float32)
        if raw is not None:
            raw[uk] += (Gm @ Xc[:, gsel]).toarray().astype(np.float32)
        np.add.at(cnt, (key, 0), 1)
        for q in range(3):
            np.add.at(cnt, (key[ph == q], 1 + q), 1)
        np.add.at(cnt, (key, 4), libc)
        own = np.zeros(len(cells)); has = tcol[tc] >= 0
        if has.any():
            own[has] = np.asarray(Xc[np.where(has)[0], tcol[tc[has]]]).ravel()
        np.add.at(cnt, (key, 5), own)
        m = tc == G_["ctrl_t"]
        craw[L] += np.asarray(Xc[np.where(m)[0]][:, gsel].sum(0)).ravel(); clib[L] += libc[m].sum()
    for nm, a in (("acc", acc), ("cnt", cnt), ("craw", craw), ("clib", clib)) + ((("raw", raw),) if raw is not None else ()):
        np.save(os.path.join(out, "parts", "%s_%s.npy" % (nm, tag)), a)
    return len(files), ncell, time.time() - t0


def main():
    raw, out, nw = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with_raw = len(sys.argv) > 4 and sys.argv[4] == "raw"     # also accumulate raw UMI sums (same keys)
    os.makedirs(out + "/parts", exist_ok=True)
    S = setup(raw)
    S["with_raw"] = with_raw
    P = phase_sets(S, out)
    mu = np.load(out + "/ctrl_mean_prepass.npy")
    gpos = {g: k for k, g in enumerate(S["genes"])}
    gsel = sorted(set(np.where(mu >= MIN_MU)[0].tolist()) | {gpos[t] for t in S["targets"] if t in gpos})
    S["gsel"] = np.array(gsel)
    ft, names, *_ = read_lane(S["files"][0])
    guides = names[S["guide_rows"]]
    par = np.zeros(len(guides), np.int64)
    for t in range(len(S["targets"])):
        ix = np.where(S["guide_target_idx"] == t)[0]
        for r, k in enumerate(ix[np.argsort(guides[ix])]):
            par[k] = r % 2
    S["guide_par"] = par
    print("lanes", len(S["files"]), "genes kept", len(gsel), "targets", len(S["targets"]),
          "guides", len(guides), "parity-1 share", float(par.mean()), flush=True)
    chunks = [(S["files"][k::nw], out, "%02d" % k) for k in range(nw) if S["files"][k::nw]]
    with Pool(len(chunks), initializer=init, initargs=(S, P)) as pool:
        for nf, nc, dt in pool.imap_unordered(do_chunk, chunks):
            print("chunk", nf, "lanes", nc, "cells", "%.0fs" % dt, flush=True)
    T, Gs = len(S["targets"]), len(gsel)
    parts = sorted(glob.glob(out + "/parts/acc_*.npy"))
    acc = np.zeros((T * 4, Gs)); cnt = np.zeros((T * 4, 6)); craw = 0; clib = 0
    for f in parts:
        acc += np.load(f); cnt += np.load(f.replace("acc_", "cnt_")); craw = craw + np.load(f.replace("acc_", "craw_"))
        clib = clib + np.load(f.replace("acc_", "clib_"))
    np.save(out + "/sums64.npy", acc); np.save(out + "/counts.npy", cnt); np.save(out + "/ctrl_raw.npy", craw)
    np.save(out + "/ctrl_lib.npy", clib)
    if with_raw:
        rawsum = np.zeros((T * 4, Gs))
        for f in parts:
            rawsum += np.load(f.replace("acc_", "raw_"))
        np.save(out + "/raw_sums.npy", rawsum)
    for f in glob.glob(out + "/parts/*.npy"):
        os.remove(f)
    json.dump(dict(targets=S["targets"], genes=[S["genes"][j] for j in gsel], lanes=[os.path.basename(f) for f in S["files"]],
                   ctrl="non-targeting", key="target * 4 + guide_parity * 2 + half",
                   counts_cols=["n", "G1", "S", "G2M", "lib_sum", "own_target_raw_sum"], min_mu=MIN_MU,
                   guide_call=dict(min_top=MIN_TOP, frac=FRAC)), open(out + "/index.json", "w"))
    print("saved; cells", int(cnt[:, 0].sum()), flush=True)


if __name__ == "__main__":
    main()
