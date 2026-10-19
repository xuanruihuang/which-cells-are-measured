"""K562 side of the hESC-vs-K562 same-scale test (A100; fate-layer paper, 2026-10-18). Pan et al. 2026 report that
perturbation effects are weaker in hESC than in K562 even for well-knocked-down genes. To compare on a depth-free
scale, both sides are aggregated as RAW UMI sums, so pseudobulk log fold changes can be formed.
For each K562 screen (day 6 essential = Replogle_K562essential_wei2025.h5ad; day 8 genome-wide part 1 =
K562_gwps_raw_singlecell_01.h5ad), cells of the targets shared with the Pan library (>= 20 cells) and of the controls:
per (target, random half): raw UMI sums over the genes shared with the Pan gene list, cell number, library sum (all
genes of the file), own-target raw sum; half = crc32(cell barcode) & 1.
Usage: k562_samescale_extract.py <pan_agg_guide dir (index.json)> <out_dir> <label:h5ad> [<label:h5ad> ...]. Exploratory."""
import os, sys, json, zlib
import numpy as np
import h5py
from screen_io import obs_info, read_rows, CTRL_NAMES

MIN_CELLS = 20


def barcodes(path):
    with h5py.File(path) as f:
        o = f["obs"]
        key = o.attrs.get("_index", "_index") if hasattr(o, "attrs") else "_index"
        if isinstance(key, bytes):
            key = key.decode()
        if key not in o:
            key = [k for k in o.keys() if k in ("_index", "index", "cell_barcode", "barcode")][0]
        v = o[key][:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])


def main():
    pdir, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    pidx = json.load(open(pdir + "/index.json")); ptargets, pgenes = set(pidx["targets"]), pidx["genes"]
    for spec in sys.argv[3:]:
        lab, path = spec.split(":", 1)
        codes, cats, gem, genes = obs_info(path)
        ctrl = [c for c in CTRL_NAMES if c in cats][0]
        n = np.bincount(codes, minlength=len(cats))
        keep = sorted(c for c in cats if c in ptargets and n[cats.index(c)] >= MIN_CELLS)
        tl = keep + [ctrl]; tpos = {c: k for k, c in enumerate(tl)}
        cmap = np.full(len(cats), -1); cmap[[cats.index(c) for c in tl]] = np.arange(len(tl))
        rows = np.where(cmap[codes] >= 0)[0]
        gix = {g: j for j, g in enumerate(genes)}
        common = [g for g in pgenes if g in gix]; ccol = np.array([gix[g] for g in common])
        print(lab, "cells", len(rows), "targets", len(keep), "genes in file", len(genes), "shared genes", len(common), flush=True)
        X = read_rows(path, rows, np.arange(len(genes)))
        lib = X.sum(1); grp = cmap[codes[rows]]
        bc = barcodes(path)[rows]
        half = np.array([zlib.crc32(b.encode()) & 1 for b in bc])
        key = grp * 2 + half; K = len(tl) * 2
        rawsum = np.zeros((K, len(common))); l1p = np.zeros((K, len(common)))
        for k in range(K):
            m = key == k
            if m.any():
                Xm = X[m]
                rawsum[k] = Xm[:, ccol].sum(0)
                l1p[k] = np.log1p(Xm[:, ccol] / lib[m][:, None] * 1e4).sum(0)   # per-cell log1p CP10K, summed
        ncell = np.bincount(key, minlength=K); libsum = np.bincount(key, weights=lib, minlength=K)
        own = np.zeros(K)
        for t, c in enumerate(tl[:-1]):
            if c in gix:
                for h in range(2):
                    m = key == t * 2 + h
                    own[t * 2 + h] = X[m, gix[c]].sum()
        np.savez_compressed(out + "/%s.npz" % lab, raw=rawsum.astype(np.float32), l1p=l1p.astype(np.float32), n=ncell, lib=libsum, own=own,
                            targets=np.array(tl), genes=np.array(common))
        print(lab, "saved", flush=True)
        del X


if __name__ == "__main__":
    main()
