"""Unified fate budget, hESC side (A100; fate paper P1; expectations T1-T5 pre-stated in notes/fate_paper_plan.md §3,
commit 1f1901b). Writes the SAME layout as fate_budget_extract.py so that fate_budget.py analyses hESC together with
K562 / RPE1 / Jurkat / HepG2:
  hESC_raw.npy (targets x 2 parity x 2 half x 3 phase x 3 group x genes), hESC_cnt.npy (same keys x
  [n, lib, own, expected]), hESC_null_raw.npy / hESC_null_cnt.npy (targets x 2 parity x 2 half x ...), hESC_index.json.
Pan et al. 2026 hESC CRISPRi (GSE295214, 85 10x lanes). Cell calls, guide parity, phase sets and gene positions come
from the pan_extract / pan_extract_guide pipeline, so the numbers stay comparable with findings bb, bc, bi:
  parity   rank of the called guide among the target's guides, mod 2
  half     crc32(lane name + barcode) & 1 (as pan_extract_guide.py)
  phase    Tirosh half-1 S / G2M vs 10 expression-matched controls, seeds 7 and 100, bins from the saved control
           pre-pass mean (pan_agg_guide_raw/ctrl_mean_prepass.npy)
  group    own-target count vs expected (lambda_t x cell library), Poisson log LR with RKD, thresholds +-log 9;
           0 = knocked down, 1 = escaper, 2 = undetermined (or target gene not measured); controls stored as group 1
Genes: control >= MIN_CP10K CP10K in the previous aggregate, plus every target gene present (same rule as
fate_budget_extract.py). RKD = median knockdown residual of the non-essential tier (DepMap > -0.2), floored at 0.05,
computed from the previous aggregate's own-target and library sums.
Thinning null: control cells with the candidate target gene's count thinned by Binomial(c, RKD), classified by the same
rule, knocked-down-like subset accumulated per (target gene, parity, half).
Usage: pan_budget_extract.py <raw_dir> <pan_agg_guide_raw dir> <depmap csv> <out_dir> <workers>. Exploratory."""
import os, sys, glob, json, time, zlib
import numpy as np
from multiprocessing import Pool
from scipy.sparse import csr_matrix
from pan_extract import read_lane, setup, S_GENES, G2M
from pan_extract_guide import lane_cells2

MIN_CP10K = 0.5
L9 = np.log(9.0)
G_ = {}


def essentiality(path):
    with open(path) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        rows = [np.array([float(x) if x else np.nan for x in line.rstrip("\n").split(",")[1:]]) for line in f]
    return dict(zip(sym, np.nanmedian(np.array(rows), 0)))


def classify(c, mu, rkd):
    llr = -(1 - rkd) * mu + c * np.log(1 / rkd)
    return np.where(llr < -L9, 0, np.where(llr > L9, 1, 2))


def init(S):
    G_.update(S)


def do_chunk(args):
    files, out, tag = args
    T = len(G_["targets"]); gsel = G_["gsel"]; G = len(gsel); ct = G_["ctrl_t"]
    tcol, lam, rkd = G_["tcol"], G_["lam"], G_["rkd"]
    cand = G_["cand"]; par_all = G_["guide_par"]; gti = G_["guide_target_idx"]
    raw = np.zeros((T, 2, 2, 3, 3, G), np.float32); cnt = np.zeros((T, 2, 2, 3, 3, 4))
    nraw = np.zeros((T, 2, 2, G), np.float32); ncnt = np.zeros((T, 2, 2, 2))
    rngb = np.random.default_rng(11)
    t0 = time.time(); ncell = 0
    for fn in files:
        topg, lib, X, bc = lane_cells2(fn, G_["gene_rows"], G_["guide_rows"])
        cells = np.where(topg >= 0)[0]; ncell += len(cells)
        g = topg[cells]; ti = gti[g]; par = par_all[g]
        Xf = X[cells]; libc = lib[cells]
        Y = Xf.multiply(1e4 / np.maximum(libc, 1)[:, None]).tocsr(); Y.data = np.log1p(Y.data)
        scS = np.asarray(Y[:, G_["iS"]].mean(1)).ravel() - np.asarray(Y[:, G_["cS"]].mean(1)).ravel()
        scG = np.asarray(Y[:, G_["iG"]].mean(1)).ravel() - np.asarray(Y[:, G_["cG"]].mean(1)).ravel()
        ph = np.where((scS <= 0) & (scG <= 0), 0, np.where(scS >= scG, 1, 2))
        del Y
        lane = os.path.basename(fn)
        half = np.array([zlib.crc32((lane + b).encode()) & 1 for b in bc[cells]])
        Xs = Xf[:, gsel].tocsr()
        own = np.zeros(len(cells)); muc = np.zeros(len(cells)); has = tcol[ti] >= 0
        if has.any():
            ih = np.where(has)[0]
            own[ih] = np.asarray(Xs[ih, tcol[ti[ih]]]).ravel()
            muc[ih] = lam[tcol[ti[ih]]] * libc[ih]
        grp = np.where(has, classify(own, muc, rkd), 2)
        isc = ti == ct; grp[isc] = 1
        key = np.ravel_multi_index((ti, par, half, ph, grp), (T, 2, 2, 3, 3))
        uk, inv = np.unique(key, return_inverse=True)
        Gm = csr_matrix((np.ones(len(key), np.float32), (inv, np.arange(len(key)))), shape=(len(uk), len(key)))
        raw.reshape(-1, G)[uk] += (Gm @ Xs).toarray().astype(np.float32)
        cf = cnt.reshape(-1, 4)
        for q, col in enumerate((np.ones(len(key)), libc, own, muc)):
            cf[:, q] += np.bincount(key, weights=col, minlength=T * 36)
        for p in (0, 1):
            for h in (0, 1):
                m = np.where(isc & (par == p) & (half == h))[0]
                if not len(m):
                    continue
                Xm = Xs[m]; lm = libc[m]
                O = Xm[:, tcol[cand]].toarray(); MU = lam[tcol[cand]][None, :] * lm[:, None]
                sel = csr_matrix(classify(rngb.binomial(O.astype(np.int64), rkd), MU, rkd) == 0, dtype=np.float32)
                nraw[cand, p, h] += (sel.T @ Xm).toarray().astype(np.float32)
                ncnt[cand, p, h, 0] += np.asarray(sel.sum(0)).ravel()
                ncnt[cand, p, h, 1] += np.asarray(sel.T @ lm).ravel()
    for nm, a in (("raw", raw), ("cnt", cnt), ("nraw", nraw), ("ncnt", ncnt)):
        np.save(os.path.join(out, "parts", "%s_%s.npy" % (nm, tag)), a)
    return len(files), ncell, time.time() - t0


def main():
    rawdir, prev, dep, out, nw = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    os.makedirs(out + "/parts", exist_ok=True)
    S = setup(rawdir)
    pidx = json.load(open(prev + "/index.json"))
    pgenes = pidx["genes"]; assert S["targets"] == pidx["targets"]
    gpos = {g: k for k, g in enumerate(S["genes"])}
    craw, clib = np.load(prev + "/ctrl_raw.npy"), np.load(prev + "/ctrl_lib.npy")
    lam_prev = craw.sum(0) / clib.sum()
    cp10k = lam_prev * 1e4
    keep_names = set(np.array(pgenes)[cp10k >= MIN_CP10K].tolist()) | {t for t in S["targets"] if t in gpos}
    gsel = np.array(sorted(gpos[g] for g in keep_names if g in gpos))
    gkeep = [S["genes"][j] for j in gsel]
    spos = {g: q for q, g in enumerate(gkeep)}
    prevpos = {g: k for k, g in enumerate(pgenes)}
    lam = np.array([lam_prev[prevpos[g]] if g in prevpos else 0.0 for g in gkeep])
    S["gsel"] = gsel; S["lam"] = lam
    S["tcol"] = np.array([spos.get(t, -1) for t in S["targets"]])
    T = len(S["targets"]); pcnt = np.load(prev + "/counts.npy").reshape(T, 4, 6)
    ess = essentiality(dep); resid = {}
    for k, t in enumerate(S["targets"]):
        if k == S["ctrl_t"] or S["tcol"][k] < 0:
            continue
        exp = lam[S["tcol"][k]] * pcnt[k, :, 4].sum()
        if exp > 0:
            resid[t] = float(pcnt[k, :, 5].sum() / exp)
    ne = [t for t in resid if ess.get(t, np.nan) > -0.2]
    rkd = max(float(np.median([resid[t] for t in ne])) if ne else 0.2, 0.05)
    S["rkd"] = rkd
    mean_lib = pcnt[S["ctrl_t"], :, 4].sum() / pcnt[S["ctrl_t"], :, 0].sum()
    S["cand"] = np.array([k for k in range(T) if k != S["ctrl_t"] and S["tcol"][k] >= 0
                          and lam[S["tcol"][k]] * mean_lib >= 2])
    mu = np.load(prev + "/ctrl_mean_prepass.npy")
    rng = np.random.default_rng(7)
    sS = [g for g in S_GENES if g in gpos]; sS = list(rng.permutation(sS))[: len(sS) // 2]
    sG = [g for g in G2M if g in gpos]; sG = list(rng.permutation(sG))[: len(sG) // 2]
    S["iS"] = np.array([gpos[g] for g in sS]); S["iG"] = np.array([gpos[g] for g in sG])
    bins = np.digitize(mu, np.quantile(mu, np.linspace(0, 1, 26)[1:-1])); r = np.random.default_rng(100)
    S["cS"], S["cG"] = (np.concatenate([r.choice(np.where(bins == bins[j])[0], 10, replace=False) for j in ix])
                        for ix in (S["iS"], S["iG"]))
    ft, names, *_ = read_lane(S["files"][0])
    guides = names[S["guide_rows"]]
    par = np.zeros(len(guides), np.int64)
    for t in range(T):
        ix = np.where(S["guide_target_idx"] == t)[0]
        for q, k in enumerate(ix[np.argsort(guides[ix])]):
            par[k] = q % 2
    S["guide_par"] = par
    print("lanes", len(S["files"]), "genes kept", len(gsel), "targets", T, "RKD", round(rkd, 3),
          "non-essential targets", len(ne), "null candidates", len(S["cand"]), flush=True)
    chunks = [(S["files"][k::nw], out, "%02d" % k) for k in range(nw) if S["files"][k::nw]]
    with Pool(len(chunks), initializer=init, initargs=(S,)) as pool:
        for nf, nc, dt in pool.imap_unordered(do_chunk, chunks):
            print("chunk", nf, "lanes", nc, "cells", "%.0fs" % dt, flush=True)
    G = len(gsel)
    tot = {"raw": np.zeros((T, 2, 2, 3, 3, G), np.float32), "cnt": np.zeros((T, 2, 2, 3, 3, 4)),
           "nraw": np.zeros((T, 2, 2, G), np.float32), "ncnt": np.zeros((T, 2, 2, 2))}
    for f in sorted(glob.glob(out + "/parts/raw_*.npy")):
        tag = os.path.basename(f)[len("raw_"):]
        for nm in tot:
            tot[nm] += np.load(os.path.join(out, "parts", nm + "_" + tag))
    np.save(out + "/hESC_raw.npy", tot["raw"]); np.save(out + "/hESC_cnt.npy", tot["cnt"])
    np.save(out + "/hESC_null_raw.npy", tot["nraw"]); np.save(out + "/hESC_null_cnt.npy", tot["ncnt"])
    for f in glob.glob(out + "/parts/*.npy"):
        os.remove(f)
    gd = {S["targets"][t]: int((S["guide_target_idx"] == t).sum()) for t in range(T)}
    cyc = sorted({g for g in S_GENES + G2M if g in spos})
    json.dump(dict(targets=S["targets"], genes=gkeep, control="non-targeting", rkd=rkd,
                   n_cells=int(tot["cnt"][..., 0].sum()), residual=resid, cycle_genes=cyc,
                   null_targets=[S["targets"][k] for k in S["cand"]], guides_per_target=gd,
                   key="targets x parity x half x phase x group", groups="0 knocked down, 1 escaper, 2 undetermined",
                   counts_cols=["n", "lib", "own", "expected_own"], min_cp10k=MIN_CP10K),
              open(out + "/hESC_index.json", "w"))
    print("saved; cells by group", [int(tot["cnt"][..., g, 0].sum()) for g in range(3)], flush=True)


if __name__ == "__main__":
    main()
