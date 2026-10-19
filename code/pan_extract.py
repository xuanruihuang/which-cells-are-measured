"""Pan et al. 2026 hESC genome-scale CRISPRi Perturb-seq (GEO GSE295214; 85 10x lanes, filtered_feature_bc_matrix.h5
with Gene Expression + CRISPR Guide Capture) -> per (target, random half) aggregates on the A100, with the same
conventions as cd4_sc_extract.py / tahoe_extract.py, for the fate-layer test (is hESC "resilience" real, or are
strongly perturbed cells eliminated?) and for cross-context comparisons.
Usage: pan_extract.py <raw_dir with *.h5> <out_dir> <workers>
Guide call per cell: top guide UMI >= MIN_TOP and >= FRAC of all guide UMIs; target = guide name before the first "_"
("non-targeting" for controls). Library = all gene-expression UMIs; expression = log1p CP10K. Half = crc32(lane +
barcode) mod 2. Phase: Tirosh half-1 genes (seed 7) vs 10 expression-matched control genes each (25 bins of the control
mean from a pre-pass over PRE_LANES lanes); G1 if both scores <= 0, else the larger.
Outputs: sums (targets x 2 x genes, float32 accumulated, saved float64 after reduce), counts (targets x 2 x [n, G1, S,
G2M, lib_sum, own_target_raw_sum]), ctrl_raw (lanes x genes raw UMI sums of control cells) and ctrl_lib (lanes), per
(target, lane) cell counts, index.json (targets, genes, lanes)."""
import os, sys, glob, json, zlib, time, re
import numpy as np, h5py
from multiprocessing import Pool
from strata import S_GENES, G2M

MIN_TOP, FRAC, PRE_LANES = 3, 0.8, 6
G_ = {}


def read_lane(fn):
    with h5py.File(fn) as f:
        m = f["matrix"]
        ft = np.array([x.decode() for x in m["features"]["feature_type"][:]])
        names = np.array([x.decode() for x in m["features"]["name"][:]])
        shape = m["shape"][:]; ip = m["indptr"][:]; ind = m["indices"][:]; dat = m["data"][:].astype(np.float64)
        bc = np.array([x.decode() for x in m["barcodes"][:]])
    return ft, names, shape, ip, ind, dat, bc


def lane_cells(fn, gene_rows, guide_rows, guide_target_idx):
    """Return per-barcode: target index (-1 unassigned), library, CSR rows over gene features (log1p CP10K and raw)."""
    ft, names, shape, ip, ind, dat, bc = read_lane(fn)
    nb = shape[1]
    col = np.repeat(np.arange(nb), np.diff(ip))
    isg = np.zeros(shape[0], bool); isg[gene_rows] = True
    gpos = np.full(shape[0], -1, np.int64); gpos[gene_rows] = np.arange(len(gene_rows))
    gq = np.full(shape[0], -1, np.int64); gq[guide_rows] = np.arange(len(guide_rows))
    mg = isg[ind]
    lib = np.bincount(col[mg], weights=dat[mg], minlength=nb)
    # guide calls
    mq = gq[ind] >= 0
    qcol, qid, qv = col[mq], gq[ind[mq]], dat[mq]
    tot = np.bincount(qcol, weights=qv, minlength=nb)
    order = np.lexsort((-qv, qcol)); qcol_s, qid_s, qv_s = qcol[order], qid[order], qv[order]
    first = np.r_[True, qcol_s[1:] != qcol_s[:-1]]
    topv = np.zeros(nb); topg = np.full(nb, -1); topv[qcol_s[first]] = qv_s[first]; topg[qcol_s[first]] = qid_s[first]
    ok = (topv >= MIN_TOP) & (topv >= FRAC * np.maximum(tot, 1)) & (lib > 0)
    tgt = np.where(ok, guide_target_idx[np.maximum(topg, 0)], -1)
    from scipy.sparse import csr_matrix
    X = csr_matrix((dat[mg], (col[mg], gpos[ind[mg]])), shape=(nb, len(gene_rows)))
    return tgt, lib, X, bc


def setup(raw_dir):
    files = sorted(glob.glob(raw_dir + "/*.h5"))
    ft, names, *_ = read_lane(files[0])
    gene_rows = np.where(ft == "Gene Expression")[0]; guide_rows = np.where(ft == "CRISPR Guide Capture")[0]
    genes = list(names[gene_rows]); guides = names[guide_rows]
    gt = np.array([g.split("_")[0] if not g.startswith("non-targeting") else "non-targeting" for g in guides])
    targets = sorted(set(gt)); tpos = {t: k for k, t in enumerate(targets)}
    guide_target_idx = np.array([tpos[t] for t in gt])
    return dict(files=files, gene_rows=gene_rows, guide_rows=guide_rows, genes=genes, targets=targets,
                guide_target_idx=guide_target_idx, ctrl_t=tpos["non-targeting"])


def phase_sets(S, out):
    """Pre-pass over a few lanes: control mean per gene -> expression bins -> control gene sets for phase scores."""
    acc = np.zeros(len(S["genes"])); n = 0
    for fn in S["files"][:PRE_LANES]:
        tgt, lib, X, _ = lane_cells(fn, S["gene_rows"], S["guide_rows"], S["guide_target_idx"])
        m = np.where(tgt == S["ctrl_t"])[0]
        Y = X[m].multiply(1e4 / lib[m][:, None]).tocsr(); Y.data = np.log1p(Y.data)
        acc += np.asarray(Y.sum(0)).ravel(); n += len(m)
    mu = acc / max(n, 1)
    gpos = {g: k for k, g in enumerate(S["genes"])}
    rng = np.random.default_rng(7)
    sS = [g for g in S_GENES if g in gpos]; sS = list(rng.permutation(sS))[: len(sS) // 2]
    sG = [g for g in G2M if g in gpos]; sG = list(rng.permutation(sG))[: len(sG) // 2]
    iS, iG = np.array([gpos[g] for g in sS]), np.array([gpos[g] for g in sG])
    bins = np.digitize(mu, np.quantile(mu, np.linspace(0, 1, 26)[1:-1]))
    r = np.random.default_rng(100)
    cS, cG = (np.concatenate([r.choice(np.where(bins == bins[j])[0], 10, replace=False) for j in ix]) for ix in (iS, iG))
    np.save(out + "/ctrl_mean_prepass.npy", mu)
    return dict(iS=iS, iG=iG, cS=cS, cG=cG)


def init(S, P):
    G_.update(S); G_.update(P)


def do_chunk(args):
    files, out, tag = args
    T, G, NL = len(G_["targets"]), len(G_["genes"]), len(G_["files"])
    acc = np.zeros((T * 2, G), np.float32); cnt = np.zeros((T * 2, 6)); craw = np.zeros((NL, G)); clib = np.zeros(NL)
    per_lane = np.zeros((T, NL), np.int64)
    lane_of = {f: k for k, f in enumerate(G_["files"])}
    tgpos = {t: k for k, t in enumerate(G_["genes"])}
    tcol = np.array([tgpos.get(t, -1) for t in G_["targets"]])
    t0 = time.time(); ncell = 0
    for fn in files:
        L = lane_of[fn]
        tgt, lib, X, bc = lane_cells(fn, G_["gene_rows"], G_["guide_rows"], G_["guide_target_idx"])
        cells = np.where(tgt >= 0)[0]; ncell += len(cells)
        Xc = X[cells]; libc = lib[cells]; tc = tgt[cells]
        Y = Xc.multiply(1e4 / libc[:, None]).tocsr(); Y.data = np.log1p(Y.data)
        sS = np.asarray(Y[:, G_["iS"]].mean(1)).ravel() - np.asarray(Y[:, G_["cS"]].mean(1)).ravel()
        sG = np.asarray(Y[:, G_["iG"]].mean(1)).ravel() - np.asarray(Y[:, G_["cG"]].mean(1)).ravel()
        ph = np.where((sS <= 0) & (sG <= 0), 0, np.where(sS >= sG, 1, 2))
        lane_name = os.path.basename(fn)
        half = np.array([zlib.crc32((lane_name + b).encode()) & 1 for b in bc[cells]])
        key = tc * 2 + half
        from scipy.sparse import csr_matrix
        uk, inv = np.unique(key, return_inverse=True)
        Gm = csr_matrix((np.ones(len(key)), (inv, np.arange(len(key)))), shape=(len(uk), len(key)))
        acc[uk] += (Gm @ Y).toarray().astype(np.float32)
        np.add.at(cnt, (key, 0), 1)
        for q in range(3):
            np.add.at(cnt, (key[ph == q], 1 + q), 1)
        np.add.at(cnt, (key, 4), libc)
        own = np.zeros(len(cells)); has = tcol[tc] >= 0
        if has.any():
            own[has] = np.asarray(Xc[np.where(has)[0], tcol[tc[has]]]).ravel()
        np.add.at(cnt, (key, 5), own)
        np.add.at(per_lane, (tc, L), 1)
        m = tc == G_["ctrl_t"]
        craw[L] += np.asarray(Xc[np.where(m)[0]].sum(0)).ravel(); clib[L] += libc[m].sum()
    for nm, a in (("acc", acc), ("cnt", cnt), ("craw", craw), ("clib", clib), ("lane", per_lane)):
        np.save(os.path.join(out, "parts", "%s_%s.npy" % (nm, tag)), a)
    return len(files), ncell, time.time() - t0


def main():
    raw, out, nw = sys.argv[1], sys.argv[2], int(sys.argv[3])
    os.makedirs(out + "/parts", exist_ok=True)
    S = setup(raw)
    P = phase_sets(S, out)
    print("lanes", len(S["files"]), "genes", len(S["genes"]), "targets", len(S["targets"]), flush=True)
    chunks = [(S["files"][k::nw], out, "%02d" % k) for k in range(nw) if S["files"][k::nw]]
    with Pool(len(chunks), initializer=init, initargs=(S, P)) as pool:
        for nf, nc, dt in pool.imap_unordered(do_chunk, chunks):
            print("chunk", nf, "lanes", nc, "cells", "%.0fs" % dt, flush=True)
    T, G = len(S["targets"]), len(S["genes"])
    parts = sorted(glob.glob(out + "/parts/acc_*.npy"))
    acc = np.zeros((T * 2, G)); cnt = np.zeros((T * 2, 6)); craw = 0; clib = 0; lane = 0
    for f in parts:
        acc += np.load(f); cnt += np.load(f.replace("acc_", "cnt_")); craw = craw + np.load(f.replace("acc_", "craw_"))
        clib = clib + np.load(f.replace("acc_", "clib_")); lane = lane + np.load(f.replace("acc_", "lane_"))
    np.save(out + "/sums64.npy", acc); np.save(out + "/counts.npy", cnt); np.save(out + "/ctrl_raw.npy", craw)
    np.save(out + "/ctrl_lib.npy", clib); np.save(out + "/cells_per_lane.npy", lane)
    for f in glob.glob(out + "/parts/*.npy"):
        os.remove(f)
    json.dump(dict(targets=S["targets"], genes=S["genes"], lanes=[os.path.basename(f) for f in S["files"]],
                   ctrl="non-targeting", key="target * 2 + half", counts_cols=["n", "G1", "S", "G2M", "lib_sum", "own_target_raw_sum"],
                   guide_call=dict(min_top=MIN_TOP, frac=FRAC)), open(out + "/index.json", "w"))
    print("saved; cells", int(cnt[:, 0].sum()), flush=True)


if __name__ == "__main__":
    main()
