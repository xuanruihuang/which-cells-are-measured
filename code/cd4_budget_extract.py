"""Unified fate budget, CD4 T side (A100; fate paper P1, second eliminate-type context). Writes the SAME layout as
fate_budget_extract.py / pan_budget_extract.py so that fate_budget.py analyses it with the other five contexts:
  CD4_<tag>_raw.npy (targets x 2 parity x 2 half x 3 phase x 3 group x genes), CD4_<tag>_cnt.npy (same keys x
  [n, lib, own, expected]), CD4_<tag>_null_raw.npy / _null_cnt.npy (targets x 2 parity x 2 half x ...), _index.json.
Zhu et al. 2026 CD4 T genome-scale CRISPRi, one donor-condition .assigned_guide.h5ad (CSR), streamed in row chunks.
Cells: low_quality dropped; controls = guide_type containing "non"; targets = every target with DepMap median gene effect
< -0.2 plus a fixed random sample (seed 5) of N_NONESS non-essential targets, each with >= MIN_CELLS cells.
  parity   from the guide's own name ("GENE-1" -> 1, "GENE-2" -> 0), as cd4_sc_extract_guide.name_parity
  half     crc32(cell barcode) & 1
  phase    Tirosh half-1 S / G2M vs 10 expression-matched controls (seed 7 split, bins from the sampled control mean),
           as cd4_sc_extract_guide.py
  group    own-target count vs lambda_t x cell UMI, Poisson log LR with RKD, thresholds +-log 9 (as the other contexts);
           RKD = median residual of the expressed (expected own count >= 1 per cell) non-essential sampled targets,
           from a pre-pass over sampled chunks
Genes: control >= MIN_CP10K CP10K (sampled controls) plus every kept target gene. Thinning null as the other contexts.
Usage: cd4_budget_extract.py <h5ad> <depmap csv> <out_dir> <tag> [workers]. Exploratory."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys, json, time, zlib
import numpy as np, h5py
from multiprocessing import Pool
from scipy.sparse import csr_matrix

MIN_CELLS, N_NONESS, MIN_CP10K, CHUNK, N_SAMPLE = 20, 600, 0.5, 4000, 150
L9 = np.log(9.0)
S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1",
           "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3",
           "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1",
           "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
G2M = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO",
       "CENPF", "TACC3", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1",
       "KIF20B", "HJURP", "CDCA3", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8",
       "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3",
       "CBX5", "CENPA"]
G_ = {}


def dec(a):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def cat(o, k):
    return o[k]["codes"][:], dec(o[k]["categories"][:])


def read_rows(f, r0, r1, G):
    ip = f["X"]["indptr"][r0:r1 + 1]
    a, b = int(ip[0]), int(ip[-1])
    X = csr_matrix((f["X"]["data"][a:b].astype(np.float32), f["X"]["indices"][a:b].astype(np.int32), ip - a), shape=(r1 - r0, G))
    X.sum_duplicates()
    return X


def essentiality(path):
    with open(path) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        rows = [np.array([float(x) if x else np.nan for x in line.rstrip("\n").split(",")[1:]]) for line in f]
    return dict(zip(sym, np.nanmedian(np.array(rows), 0)))


def classify(c, mu, rkd):
    llr = -(1 - rkd) * mu + c * np.log(1 / rkd)
    return np.where(llr < -L9, 0, np.where(llr > L9, 1, 2))


def name_parity(names):
    out = np.zeros(len(names), np.int64)
    for i, x in enumerate(names):
        a = x.rsplit("-", 1)
        out[i] = int(a[1]) % 2 if len(a) == 2 and a[1].isdigit() else 0
    return out


def phases(Xd_small, umi):
    sm = np.log1p(Xd_small / umi[:, None] * 1e4)
    sS = sm[:, G_["iS"]].mean(1) - sm[:, G_["iSc"]].mean(1); sG = sm[:, G_["iG"]].mean(1) - sm[:, G_["iGc"]].mean(1)
    return np.where((sS <= 0) & (sG <= 0), 0, np.where(sS >= sG, 1, 2))


def worker(args):
    wid, r0w, r1w = args
    g = G_; T, G = g["T"], len(g["gsel"])
    raw = np.zeros((T * 36, G), np.float32); cnt = np.zeros((T * 36, 4))
    nraw = np.zeros((T, 2, 2, G), np.float32); ncnt = np.zeros((T, 2, 2, 2))
    rngb = np.random.default_rng(11 + wid)
    cand, tcol, lam, rkd = g["cand"], g["tcol"], g["lam"], g["rkd"]
    with h5py.File(g["fn"], "r") as f:
        for r0 in range(r0w, r1w, CHUNK):
            r1 = min(r0 + CHUNK, r1w)
            ti = g["ti"][r0:r1]
            v = np.where(ti >= 0)[0]
            if not len(v):
                continue
            X = read_rows(f, r0, r1, g["Gfull"])[v]
            umi = np.asarray(X.sum(1)).ravel(); ok = umi > 0
            v, X, umi = v[ok], X[np.where(ok)[0]], umi[ok]
            ti_v = ti[v]
            ph = phases(X[:, g["pcols"]].toarray(), umi)
            Xs = X[:, g["gsel"]].tocsr()
            has = tcol[ti_v] >= 0
            own = np.zeros(len(v)); muc = np.zeros(len(v))
            if has.any():
                ih = np.where(has)[0]
                own[ih] = np.asarray(Xs[ih, tcol[ti_v[ih]]]).ravel()
                muc[ih] = lam[tcol[ti_v[ih]]] * umi[ih]
            grp = np.where(has, classify(own, muc, rkd), 2)
            isc = ti_v == T - 1; grp[isc] = 1
            par = g["par"][r0:r1][v]; half = g["half"][r0:r1][v]
            key = (((ti_v * 2 + par) * 2 + half) * 3 + ph) * 3 + grp
            uk, inv = np.unique(key, return_inverse=True)
            Gm = csr_matrix((np.ones(len(key), np.float32), (inv, np.arange(len(key)))), shape=(len(uk), len(key)))
            raw[uk] += (Gm @ Xs).toarray().astype(np.float32)
            for q, col in enumerate((np.ones(len(key)), umi, own, muc)):
                cnt[:, q] += np.bincount(key, weights=col, minlength=T * 36)
            for p in (0, 1):
                for h in (0, 1):
                    m = np.where(isc & (par == p) & (half == h))[0]
                    if not len(m):
                        continue
                    Xm = Xs[m]; lm = umi[m]
                    O = Xm[:, tcol[cand]].toarray(); MU = lam[tcol[cand]][None, :] * lm[:, None]
                    sel = csr_matrix(classify(rngb.binomial(O.astype(np.int64), rkd), MU, rkd) == 0, dtype=np.float32)
                    nraw[cand, p, h] += (sel.T @ Xm).toarray().astype(np.float32)
                    ncnt[cand, p, h, 0] += np.asarray(sel.sum(0)).ravel(); ncnt[cand, p, h, 1] += np.asarray(sel.T @ lm).ravel()
    for nm, a in (("raw", raw), ("cnt", cnt), ("nraw", nraw), ("ncnt", ncnt)):
        np.save("%s/parts/%s_%03d.npy" % (g["out"], nm, wid), a)
    return wid


def main():
    fn, dep, out, tag = sys.argv[1:5]; nw = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    os.makedirs(out + "/parts", exist_ok=True); t0 = time.time()
    ess = essentiality(dep)
    with h5py.File(fn, "r") as f:
        o = f["obs"]
        pc, pcats = cat(o, "perturbed_gene_name"); gc, gcats = cat(o, "guide_type"); gid, gidcats = cat(o, "guide_id")
        lq = o["low_quality"][:].astype(bool)
        ik = o.attrs.get("_index", "_index"); ik = ik.decode() if isinstance(ik, bytes) else ik
        bcs = dec(o[ik][:])
        genes = dec(f["var"]["gene_name"][:])
        N, Gfull = (int(x) for x in f["X"].attrs["shape"])
    gpos = {}
    for k, x in enumerate(genes):
        gpos.setdefault(x, k)
    ctrl = np.isin(gc, [k for k, x in enumerate(gcats) if "non" in x.lower()]) & ~lq
    tgt_ok = (pc >= 0) & ~lq & ~ctrl
    ncell = np.bincount(pc[tgt_ok], minlength=len(pcats))
    ess_t = [k for k, x in enumerate(pcats) if ncell[k] >= MIN_CELLS and ess.get(x, np.nan) < -0.2]
    nonp = [k for k, x in enumerate(pcats) if ncell[k] >= MIN_CELLS and ess.get(x, np.nan) > -0.2]
    nonp = sorted(np.random.default_rng(5).choice(nonp, min(N_NONESS, len(nonp)), replace=False).tolist())
    keep_cats = sorted(ess_t + nonp); tl = [pcats[k] for k in keep_cats] + ["non-targeting"]; T = len(tl)
    cmap = np.full(len(pcats), -1); cmap[keep_cats] = np.arange(T - 1)
    ti = np.where(tgt_ok, cmap[np.maximum(pc, 0)], -1); ti[ctrl] = T - 1
    par = name_parity(gidcats); par = np.where(gid >= 0, par[np.maximum(gid, 0)], 0).astype(np.int64)
    half = np.array([zlib.crc32(b.encode()) & 1 for b in bcs], np.int64)
    print("cells", N, "genes", Gfull, "targets kept", T - 1, "(essential", len(ess_t), "+ non-essential sample", len(nonp), ")",
          "controls", int(ctrl.sum()), "%.0fs" % (time.time() - t0), flush=True)
    # pre-pass over sampled chunks: control rates, control log mean (phase bins), own-target residuals
    rng = np.random.default_rng(3)
    starts = np.sort(rng.choice(np.arange(0, N - CHUNK, CHUNK), N_SAMPLE, replace=False))
    craw = np.zeros(Gfull); cumi = 0.0; clog = np.zeros(Gfull); nc = 0
    own_s = np.zeros(T); exp_umi = np.zeros(T)
    t_gene = np.array([gpos.get(t, -1) for t in tl])
    with h5py.File(fn, "r") as f:
        for r0 in starts:
            X = read_rows(f, r0, r0 + CHUNK, Gfull); umi = np.asarray(X.sum(1)).ravel()
            tic = ti[r0:r0 + CHUNK]
            m = np.where((tic == T - 1) & (umi > 0))[0]
            if len(m):
                Xc = X[m]; craw += np.asarray(Xc.sum(0)).ravel(); cumi += umi[m].sum()
                L = Xc.multiply(1e4 / umi[m][:, None]).tocsr(); L.data = np.log1p(L.data); clog += np.asarray(L.sum(0)).ravel(); nc += len(m)
            m = np.where((tic >= 0) & (tic < T - 1) & (umi > 0))[0]
            for i in m:
                j = t_gene[tic[i]]
                if j >= 0:
                    own_s[tic[i]] += X[i, j]; exp_umi[tic[i]] += umi[i]
    lam_full = craw / cumi; mu0 = clog / nc
    gsel = np.array(sorted(set(np.where(lam_full * 1e4 >= MIN_CP10K)[0].tolist()) | {j for j in t_gene[:-1] if j >= 0}))
    spos = {j: q for q, j in enumerate(gsel)}
    lam = lam_full[gsel]
    tcol = np.array([spos.get(j, -1) if j >= 0 else -1 for j in t_gene])
    resid = {tl[k]: float(own_s[k] / (lam_full[t_gene[k]] * exp_umi[k])) for k in range(T - 1)
             if t_gene[k] >= 0 and exp_umi[k] > 0 and lam_full[t_gene[k]] > 0}
    mean_umi = cumi / max(nc, 1)
    # RKD from expressed non-essential targets only (expected own count >= 1 per control cell); unexpressed targets give
    # residual ~0 and pulled the median to the floor in the first launch (10-18)
    ne = [t for t in resid if ess.get(t, np.nan) > -0.2 and lam_full[t_gene[tl.index(t)]] * mean_umi >= 1]
    rkd = max(float(np.median([resid[t] for t in ne])) if ne else 0.2, 0.05)
    cand = np.array([k for k in range(T - 1) if tcol[k] >= 0 and lam[tcol[k]] * mean_umi >= 2])
    r7 = np.random.default_rng(7)
    sp = lambda gl: (lambda g: g[: len(g) // 2])(list(r7.permutation([x for x in gl if x in gpos])))
    s1, g1 = sp(S_GENES), sp(G2M)
    bins = np.digitize(mu0, np.quantile(mu0, np.linspace(0, 1, 26)[1:-1]))
    pick = lambda st: np.concatenate([r7.choice(np.where(bins == bins[gpos[x]])[0], 10, replace=False) for x in st])
    sets = dict(S=np.array([gpos[x] for x in s1]), Sc=pick(s1), G=np.array([gpos[x] for x in g1]), Gc=pick(g1))
    pcols = np.unique(np.concatenate(list(sets.values()))); cpos = {c: k for k, c in enumerate(pcols)}
    G_.update(fn=fn, out=out, T=T, Gfull=Gfull, gsel=gsel, ti=ti, par=par, half=half, tcol=tcol, lam=lam, rkd=rkd, cand=cand,
              pcols=pcols, **{"i" + k: np.array([cpos[c] for c in v]) for k, v in sets.items()})
    print("pre-pass: sampled controls", nc, "genes kept", len(gsel), "RKD", round(rkd, 3), "non-essential residuals", len(ne),
          "null candidates", len(cand), "%.0fs" % (time.time() - t0), flush=True)
    edges = np.linspace(0, N, nw + 1).astype(int)
    with Pool(nw) as pool:
        for wid in pool.imap_unordered(worker, [(w, edges[w], edges[w + 1]) for w in range(nw)]):
            print("worker", wid, "done %.0fs" % (time.time() - t0), flush=True)
    G = len(gsel)
    tot = {"raw": np.zeros((T * 36, G), np.float32), "cnt": np.zeros((T * 36, 4)), "nraw": np.zeros((T, 2, 2, G), np.float32), "ncnt": np.zeros((T, 2, 2, 2))}
    for w in range(nw):
        for nm in tot:
            tot[nm] += np.load("%s/parts/%s_%03d.npy" % (out, nm, w))
    lab = "CD4_" + tag
    np.save("%s/%s_raw.npy" % (out, lab), tot["raw"].reshape(T, 2, 2, 3, 3, G)); np.save("%s/%s_cnt.npy" % (out, lab), tot["cnt"].reshape(T, 2, 2, 3, 3, 4))
    np.save("%s/%s_null_raw.npy" % (out, lab), tot["nraw"]); np.save("%s/%s_null_cnt.npy" % (out, lab), tot["ncnt"])
    for w in range(nw):
        for nm in tot:
            os.remove("%s/parts/%s_%03d.npy" % (out, nm, w))
    gl_pt = {}
    for k in keep_cats:
        gl_pt[pcats[k]] = int(len(np.unique(gid[(pc == k) & ~lq])))
    gnames = [genes[j] for j in gsel]
    json.dump(dict(targets=tl, genes=gnames, control="non-targeting", rkd=rkd, n_cells=int(tot["cnt"][:, 0].sum()), residual=resid,
                   cycle_genes=sorted({x for x in S_GENES + G2M if x in set(gnames)}), null_targets=[tl[k] for k in cand],
                   guides_per_target=gl_pt, target_selection="DepMap < -0.2 all, plus %d sampled non-essential (seed 5)" % N_NONESS,
                   key="targets x parity x half x phase x group", groups="0 knocked down, 1 escaper, 2 undetermined",
                   counts_cols=["n", "lib", "own", "expected_own"], min_cp10k=MIN_CP10K, rkd_from="sampled pre-pass"),
              open("%s/%s_index.json" % (out, lab), "w"))
    print("saved", lab, "cells", int(tot["cnt"][:, 0].sum()), "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
