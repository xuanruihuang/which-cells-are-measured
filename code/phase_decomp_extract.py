"""Phase-stratified pseudobulk for the composition decomposition (A100; fate paper C2; expectations S1-S4 pre-stated in
notes/fate_paper_plan.md §3, commit e1f14f4). For each Replogle-format raw single-cell file (label:path), cells of the
targets with >= 20 cells in every file and of the non-targeting controls:
  phase per cell from Tirosh half-1 S / G2M genes vs 10 expression-matched control genes each (seeds 7 and 100, as
  pan_extract.py and fate_h1.py; bins from this file's control cells), 0 = G1, 1 = S, 2 = G2M;
  random half = crc32(cell barcode) & 1;
  raw UMI sums over the genes shared by all files, per (target, half, phase), with cell numbers and library sums.
Usage: phase_decomp_extract.py <out_dir> <label:h5ad> [<label:h5ad> ...]. Exploratory."""
import os, sys, zlib
import numpy as np
from scipy.sparse import csr_matrix
from screen_io import obs_info, read_rows, CTRL_NAMES
from strata import S_GENES, G2M
from k562_samescale_extract import barcodes

MIN_CELLS = 20


def first_index(genes):
    gix = {}
    for j, g in enumerate(genes):
        gix.setdefault(str(g), j)
    return gix


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    info = {}
    for spec in sys.argv[2:]:
        lab, path = spec.split(":", 1)
        codes, cats, gem, genes = obs_info(path)
        info[lab] = (path, codes, [str(c) for c in cats], [str(g) for g in genes])
    tsets = []
    for lab, (path, codes, cats, genes) in info.items():
        n = np.bincount(codes, minlength=len(cats))
        tsets.append({c for k, c in enumerate(cats) if n[k] >= MIN_CELLS and c not in CTRL_NAMES})
    shared = sorted(set.intersection(*tsets))
    gshared = sorted(set.intersection(*[set(v[3]) for v in info.values()]))
    print("shared targets", len(shared), "shared genes", len(gshared), flush=True)
    for lab, (path, codes, cats, genes) in info.items():
        ctrl = [c for c in CTRL_NAMES if c in cats][0]
        tl = shared + [ctrl]; K = len(tl)
        cmap = np.full(len(cats), -1); cmap[[cats.index(c) for c in tl]] = np.arange(K)
        rows = np.where(cmap[codes] >= 0)[0]
        gix = first_index(genes)
        X = read_rows(path, rows, np.arange(len(genes)))
        lib = X.sum(1).astype(np.float64); grp = cmap[codes[rows]]
        half = np.array([zlib.crc32(b.encode()) & 1 for b in barcodes(path)[rows]])
        Y = np.log1p(X / np.maximum(lib, 1)[:, None] * 1e4).astype(np.float32)
        mu = Y[grp == K - 1].mean(0)
        rng = np.random.default_rng(7)
        sS = [g for g in S_GENES if g in gix]; sS = list(rng.permutation(sS))[: len(sS) // 2]
        sG = [g for g in G2M if g in gix]; sG = list(rng.permutation(sG))[: len(sG) // 2]
        iS, iG = np.array([gix[g] for g in sS]), np.array([gix[g] for g in sG])
        bins = np.digitize(mu, np.quantile(mu, np.linspace(0, 1, 26)[1:-1])); r = np.random.default_rng(100)
        cS, cG = (np.concatenate([r.choice(np.where(bins == bins[j])[0], 10, replace=False) for j in ix]) for ix in (iS, iG))
        scS = Y[:, iS].mean(1) - Y[:, cS].mean(1); scG = Y[:, iG].mean(1) - Y[:, cG].mean(1)
        ph = np.where((scS <= 0) & (scG <= 0), 0, np.where(scS >= scG, 1, 2))
        del Y
        key = grp * 6 + half * 3 + ph; NK = K * 6
        Gm = csr_matrix((np.ones(len(key)), (key, np.arange(len(key)))), shape=(NK, len(key)))
        gcol = np.array([gix[g] for g in gshared])
        raw = np.asarray(Gm @ X[:, gcol]).astype(np.float32)
        n = np.bincount(key, minlength=NK); libs = np.bincount(key, weights=lib, minlength=NK)
        g1c = float((ph[grp == K - 1] == 0).mean())
        np.savez_compressed("%s/%s.npz" % (out, lab), raw=raw, n=n, lib=libs, targets=np.array(tl), genes=np.array(gshared))
        print(lab, "cells", len(rows), "control G1 fraction", round(g1c, 3), "phase shares", np.bincount(ph, minlength=3) / len(ph), flush=True)
        del X


if __name__ == "__main__":
    main()
