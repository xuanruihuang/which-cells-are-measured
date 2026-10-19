"""Unified fate budget, extraction for Replogle-format screens (runs on either server; fate paper P1; expectations
T1-T5 pre-stated in notes/fate_paper_plan.md §3, commit 1f1901b). Self-contained readers so the same file works on the
A100 (Replogle K562 / RPE1) and the 5090 (Nadig Jurkat / HepG2).
Per file: cells of the targets with >= MIN_CELLS cells and of the non-targeting controls. Kept genes: control mean
>= MIN_CP10K, plus every target gene present. Per cell:
  phase    Tirosh half-1 S / G2M genes vs 10 expression-matched control genes each (seeds 7 and 100, bins from the
           control cells; same rule as pan_extract.py, fate_h1.py, phase_decomp_extract.py): 0 = G1, 1 = S, 2 = G2M
  parity   rank of the called guide (sgID_AB) among the target's guides, mod 2; controls likewise
  half     crc32(cell barcode) & 1
  group    own-target count c vs expected mu = lambda_t x cell library, Poisson log LR = -(1 - RKD) mu + c log(1 / RKD):
           0 = knocked down (< -log 9), 1 = escaper (> log 9), 2 = undetermined (or target gene not measured).
           RKD = median knockdown residual of the non-essential tier of this context (DepMap > -0.2), floored at 0.05.
Thinning null (as findings bi; the tail null is unusable): control cells with the candidate target gene's count thinned
by Binomial(c, RKD), classified by the same rule, knocked-down-like subset accumulated per (target gene, parity, half).
Outputs: raw.npy (targets x 2 parity x 2 half x 3 phase x 3 group x genes), cnt.npy (same keys x [n, lib, own, mu]),
null_raw.npy / null_cnt.npy (targets x 2 x 2 x ...), ctrl_raw.npy / ctrl_cnt.npy (2 x 2 x 3 x ...), index.json.
Set FATE_TARGET_LIST=<index.json> to keep only that extraction's targets (used to pair the K562 day-8 genome-wide
screen with the day-6 essential screen as independent units; without it the day-8 arrays would be far too large).
Usage: fate_budget_extract.py <depmap csv> <out_dir> <label:h5ad> [<label:h5ad> ...]. Exploratory."""
import os, sys, json, zlib
import numpy as np
import h5py
from scipy.sparse import csr_matrix

MIN_CELLS = 20
MIN_CP10K = 0.5
BLOCK = 20000
L9 = np.log(9.0)
CTRL_NAMES = ("non-targeting", "CTRL", "control", "non_targeting")
S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1",
           "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3",
           "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1",
           "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
G2M = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO",
       "CENPF", "TACC3", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1",
       "KIF20B", "HJURP", "CDCA3", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8",
       "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3",
       "CBX5", "CENPA"]


def dec(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def obs_str(f, key):
    o = f["obs"]; n = o[key]
    if isinstance(n, h5py.Group):
        c = np.array([dec(x) for x in n["categories"][:]]); return c[n["codes"][:]]
    if "__categories" in o and key in o["__categories"]:
        c = np.array([dec(x) for x in o["__categories"][key][:]]); return c[n[:]]
    return np.array([dec(x) for x in n[:]])


def var_names(f):
    v = f["var"]
    for k in ("gene_name", "gene_symbol", "_index", "index"):
        if k in v:
            n = v[k]
            if isinstance(n, h5py.Group):
                c = np.array([dec(x) for x in n["categories"][:]]); return c[n["codes"][:]]
            if "__categories" in v and k in v["__categories"]:          # old anndata: integer codes + category table
                c = np.array([dec(x) for x in v["__categories"][k][:]]); return c[n[:]]
            return np.array([dec(x) for x in n[:]])
    raise KeyError("no gene names in var")


def matrix(f):
    return f["layers"]["counts"] if "layers" in f and "counts" in f["layers"] else f["X"]


def read_dense(path, rows):
    """Dense (len(rows) x all genes) float32, read in contiguous blocks."""
    rows = np.asarray(rows)
    with h5py.File(path) as f:
        M = matrix(f)
        if not isinstance(M, h5py.Dataset):
            raise TypeError("sparse X not supported here")
        out = np.zeros((len(rows), M.shape[1]), np.float32)
        order = np.argsort(rows); rs = rows[order]
        pos = 0
        while pos < len(rs):
            lo = rs[pos]; hi = min(lo + BLOCK, M.shape[0])
            sel = rs[pos:][rs[pos:] < hi]
            blk = M[lo:hi]
            out[order[pos:pos + len(sel)]] = blk[sel - lo]
            pos += len(sel)
    return out


def essentiality(path):
    with open(path) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        rows = [np.array([float(x) if x else np.nan for x in line.rstrip("\n").split(",")[1:]]) for line in f]
    return dict(zip(sym, np.nanmedian(np.array(rows), 0)))


def classify(c, mu, rkd):
    return np.where(-(1 - rkd) * mu + c * np.log(1 / rkd) < -L9, 0,
                    np.where(-(1 - rkd) * mu + c * np.log(1 / rkd) > L9, 1, 2))


def main():
    dep, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    ess = essentiality(dep)
    for spec in sys.argv[3:]:
        lab, path = spec.split(":", 1)
        with h5py.File(path) as f:
            tgt = obs_str(f, "gene"); guide = obs_str(f, "sgID_AB"); bc = obs_str(f, "cell_barcode")
            genes = var_names(f)
        ctrl = [c for c in CTRL_NAMES if c in set(tgt)][0]
        cnt_t = {}
        for t in tgt:
            cnt_t[t] = cnt_t.get(t, 0) + 1
        keep_t = sorted(t for t in cnt_t if t != ctrl and cnt_t[t] >= MIN_CELLS)
        only = os.environ.get("FATE_TARGET_LIST")                 # restrict to another extraction's targets
        if only:
            keep = set(json.load(open(only))["targets"])
            keep_t = [t for t in keep_t if t in keep]
            print(lab, "restricted to", len(keep_t), "targets from", only, flush=True)
        tl = keep_t + [ctrl]; T = len(tl); tpos = {t: k for k, t in enumerate(tl)}
        rows = np.where(np.isin(tgt, tl))[0]
        print(lab, "cells", len(rows), "targets", len(keep_t), flush=True)
        X = read_dense(path, rows)
        tg = tgt[rows]; gd = guide[rows]; bcs = bc[rows]
        lib = X.sum(1).astype(np.float64)
        gix = {}
        for j, g in enumerate(genes):
            gix.setdefault(str(g), j)
        isc = tg == ctrl
        cm = X[isc].sum(0) / lib[isc].sum() * 1e4
        gsel = sorted(set(np.where(cm >= MIN_CP10K)[0].tolist()) | {gix[t] for t in keep_t if t in gix})
        gsel = np.array(gsel); gkeep = [str(genes[j]) for j in gsel]
        spos = {g: q for q, g in enumerate(gkeep)}
        # phase
        Y = np.log1p(X / np.maximum(lib, 1)[:, None] * 1e4)
        mu_c = Y[isc].mean(0)
        rng = np.random.default_rng(7)
        sS = [g for g in S_GENES if g in gix]; sS = list(rng.permutation(sS))[: len(sS) // 2]
        sG = [g for g in G2M if g in gix]; sG = list(rng.permutation(sG))[: len(sG) // 2]
        iS = np.array([gix[g] for g in sS]); iG = np.array([gix[g] for g in sG])
        if len(iS) == 0 or len(iG) == 0:
            raise ValueError("no S / G2M genes matched in var names (first names: %s)" % list(genes[:5]))
        bins = np.digitize(mu_c, np.quantile(mu_c, np.linspace(0, 1, 26)[1:-1])); r = np.random.default_rng(100)
        cS, cG = (np.concatenate([r.choice(np.where(bins == bins[j])[0], 10, replace=False) for j in ix]) for ix in (iS, iG))
        scS = Y[:, iS].mean(1) - Y[:, cS].mean(1); scG = Y[:, iG].mean(1) - Y[:, cG].mean(1)
        ph = np.where((scS <= 0) & (scG <= 0), 0, np.where(scS >= scG, 1, 2))
        del Y
        # parity and half
        par = np.zeros(len(rows), np.int64)
        for t in tl:
            m = tg == t; gl = sorted(set(gd[m]))
            rank = {g: k % 2 for k, g in enumerate(gl)}
            par[m] = [rank[g] for g in gd[m]]
        half = np.array([zlib.crc32(b.encode()) & 1 for b in bcs])
        # lambda, residual per target, RKD from the non-essential tier
        lam = X[isc][:, gsel].sum(0) / lib[isc].sum()
        tcol = np.array([spos.get(t, -1) for t in tl])
        own = np.zeros(len(rows)); muc = np.zeros(len(rows))
        ti = np.array([tpos[t] for t in tg]); has = tcol[ti] >= 0
        if has.any():
            ih = np.where(has)[0]
            own[ih] = X[ih, gsel[tcol[ti[ih]]]]
            muc[ih] = lam[tcol[ti[ih]]] * lib[ih]
        resid = {}
        for k, t in enumerate(tl[:-1]):
            m = (ti == k) & has
            if m.any() and muc[m].sum() > 0:
                resid[t] = float(own[m].sum() / muc[m].sum())
        ne = [t for t in resid if ess.get(t, np.nan) > -0.2]
        rkd = max(float(np.median([resid[t] for t in ne])) if ne else 0.2, 0.05)
        grp = np.where(has, classify(own, muc, rkd), 2)
        grp[isc] = 1
        print(lab, "RKD", round(rkd, 3), "non-essential targets", len(ne), "genes kept", len(gsel),
              "phase shares", np.round(np.bincount(ph, minlength=3) / len(ph), 3), flush=True)
        # accumulate
        G = len(gsel)
        raw = np.zeros((T, 2, 2, 3, 3, G), np.float32); cnt = np.zeros((T, 2, 2, 3, 3, 4))
        key = np.ravel_multi_index((ti, par, half, ph, grp), (T, 2, 2, 3, 3))
        Xs = X[:, gsel]
        uk, inv = np.unique(key, return_inverse=True)
        Gm = csr_matrix((np.ones(len(key), np.float32), (inv, np.arange(len(key)))), shape=(len(uk), len(key)))
        acc = Gm @ Xs                                              # keys x genes
        flat = raw.reshape(-1, G); flat[uk] = acc
        cflat = cnt.reshape(-1, 4)
        for q, col in enumerate((np.ones(len(key)), lib, own, muc)):
            cflat[:, q] = np.bincount(key, weights=col, minlength=T * 36)
        # thinning null on control cells
        cand = np.array([k for k in range(T - 1) if tcol[k] >= 0 and lam[tcol[k]] * lib[isc].mean() >= 2])
        nraw = np.zeros((T, 2, 2, G), np.float32); ncnt = np.zeros((T, 2, 2, 2))
        rngb = np.random.default_rng(11)
        for p in (0, 1):
            for h in (0, 1):
                m = np.where(isc & (par == p) & (half == h))[0]
                if not len(m):
                    continue
                Om = Xs[m][:, tcol[cand]]; MUm = lam[tcol[cand]][None, :] * lib[m][:, None]
                thin = rngb.binomial(Om.astype(np.int64), rkd)
                sel = classify(thin, MUm, rkd) == 0                      # cells x cand
                nraw[cand, p, h] = sel.T.astype(np.float32) @ Xs[m]
                ncnt[cand, p, h, 0] = sel.sum(0); ncnt[cand, p, h, 1] = sel.T.astype(np.float64) @ lib[m]
        np.save("%s/%s_raw.npy" % (out, lab), raw); np.save("%s/%s_cnt.npy" % (out, lab), cnt)
        np.save("%s/%s_null_raw.npy" % (out, lab), nraw); np.save("%s/%s_null_cnt.npy" % (out, lab), ncnt)
        cyc = sorted({g for g in S_GENES + G2M if g in spos})
        json.dump(dict(targets=tl, genes=gkeep, control=ctrl, rkd=rkd, n_cells=int(len(rows)),
                       residual=resid, cycle_genes=cyc, null_targets=[tl[k] for k in cand],
                       guides_per_target={t: len(set(gd[tg == t])) for t in tl},
                       key="targets x parity x half x phase x group", groups="0 knocked down, 1 escaper, 2 undetermined",
                       counts_cols=["n", "lib", "own", "expected_own"], min_cells=MIN_CELLS, min_cp10k=MIN_CP10K),
                  open("%s/%s_index.json" % (out, lab), "w"))
        print(lab, "saved; cells by group", [int(cnt[..., g, 0].sum()) for g in range(3)], flush=True)
        del X, Xs


if __name__ == "__main__":
    main()
