"""P4 item 3, Leng 2022 descriptive check (fate paper; pre-stated in notes/fate_paper_plan.md, commit e624ade).
iPSC-derived astrocytes, CRISPRi, GSE182308: vehicle (GEM wells 1-2) and IL-1a + TNF + C1q (wells 3-4); per-cell
sgRNA counts are per TARGET (no guide identity). Per state and target:
  cells (single assignment: top target >= 5 UMI and >= 60% of the cell's sgRNA UMI), relative cells (/ median of the
  non-essential targets, DepMap > -0.2), residual = own raw / (NTC rate x library), and for the NTC cells the S / G2M
  share by the same Tirosh half-1 rule as fate_budget_extract.py plus the share of cells with MKI67 or TOP2A > 0.
Descriptive only (n = 3 essential targets). Usage: fate_leng.py <raw dir> <depmap csv> <out json>."""
import sys, glob, json, re
import numpy as np, pandas as pd, h5py
from scipy.sparse import csc_matrix
from fate_budget_extract import S_GENES, G2M, essentiality


def read10x(fn):
    m = h5py.File(fn)["matrix"]
    X = csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:])).T.tocsr()  # cells x genes
    return X, m["barcodes"][:].astype(str), m["features"]["name"][:].astype(str)


def phase(X, lib, gix, ctl_rows):
    Y = np.log1p(X.multiply(1e4 / np.maximum(lib, 1)[:, None]).tocsr().toarray()) if X.shape[1] < 3000 else None
    return Y


def main():
    raw, dep, outf = sys.argv[1:4]; ess = essentiality(dep)
    wells = sorted(glob.glob(raw + "/*_filtered_feature_bc_matrix.h5"))
    res = {"rules": __doc__, "states": {}}
    by_state = {}
    for fn in wells:
        state = "vehicle" if "_Veh_" in fn else "ITC"
        X, bcs, genes = read10x(fn)
        sg = pd.read_csv(fn.replace("_filtered_feature_bc_matrix.h5", "_sgRNAcounts.csv.gz"), index_col=0)
        cols = {c.split("-")[0]: c for c in sg.columns}
        sgv = sg.values; tot = sgv.sum(0); top = sgv.argmax(0); topv = sgv.max(0)
        ok = (topv >= 5) & (topv >= 0.6 * np.maximum(tot, 1))
        assign = {c.split("-")[0]: sg.index[top[k]] for k, c in enumerate(sg.columns) if ok[k]}
        tg = np.array([assign.get(b.split("-")[0], "") for b in bcs])
        by_state.setdefault(state, []).append((X, tg, genes))
        print(fn.split("/")[-1], "cells", len(bcs), "assigned", int((tg != "").sum()), flush=True)
    for state, parts in by_state.items():
        genes = parts[0][2]; gpos = {g: j for j, g in enumerate(genes)}
        from scipy.sparse import vstack
        X = vstack([p[0] for p in parts]).tocsr(); tg = np.concatenate([p[1] for p in parts])
        lib = np.asarray(X.sum(1)).ravel()
        isn = np.array([t.startswith("NTC") for t in tg])
        rate = np.asarray(X[isn].sum(0)).ravel() / lib[isn].sum()
        # phase rule on NTC cells (Tirosh half-1, seeds 7 / 100, expression-matched controls)
        Yc = X[isn].multiply(1e4 / lib[isn][:, None]).tocsr(); Yc.data = np.log1p(Yc.data)
        mu = np.asarray(Yc.mean(0)).ravel()
        rng = np.random.default_rng(7)
        sS = [g for g in S_GENES if g in gpos]; sS = list(rng.permutation(sS))[: len(sS) // 2]
        sG = [g for g in G2M if g in gpos]; sG = list(rng.permutation(sG))[: len(sG) // 2]
        iS, iG = np.array([gpos[g] for g in sS]), np.array([gpos[g] for g in sG])
        bins = np.digitize(mu, np.quantile(mu, np.linspace(0, 1, 26)[1:-1])); r = np.random.default_rng(100)
        cS, cG = (np.concatenate([r.choice(np.where(bins == bins[j])[0], 10, replace=False) for j in ix]) for ix in (iS, iG))
        Yd = lambda idx: np.asarray(Yc[:, idx].mean(1)).ravel()
        scS, scG = Yd(iS) - Yd(cS), Yd(iG) - Yd(cG)
        ph = np.where((scS <= 0) & (scG <= 0), 0, np.where(scS >= scG, 1, 2))
        prolif = float(np.mean((np.asarray(X[isn][:, [gpos[g] for g in ("MKI67", "TOP2A") if g in gpos]].sum(1)).ravel()) > 0))
        recs = {}
        for t in sorted(set(tg) - {""}):
            m = tg == t
            j = gpos.get(t)
            exp = rate[j] * lib[m].sum() if j is not None else 0
            recs[t] = dict(cells=int(m.sum()), depmap=float(ess.get(t, np.nan)),
                           resid=float(np.asarray(X[m][:, j].sum()) / exp) if j is not None and exp >= 5 else None)
        ne = [v["cells"] for t, v in recs.items() if not t.startswith("NTC") and np.isfinite(v["depmap"]) and v["depmap"] > -0.2]
        base = float(np.median(ne))
        for v in recs.values():
            v["rel_cells"] = v["cells"] / base
        res["states"][state] = dict(ntc_phase_shares=[float(np.mean(ph == k)) for k in range(3)],
                                    ntc_mki67_or_top2a_positive=prolif, nonessential_median_cells=base, targets=recs)
        print("==", state, "NTC phase G1/S/G2M", np.round([np.mean(ph == k) for k in range(3)], 3),
              "MKI67|TOP2A>0 %.3f" % prolif, "non-essential median cells", base, flush=True)
        for t in ("NRF1", "ARPC3", "ABCE1", "NTC_1", "NTC_2"):
            if t in recs:
                print("  ", t, json.dumps(recs[t]), flush=True)
        rs = [v["resid"] for t, v in recs.items() if v["resid"] is not None and np.isfinite(v["depmap"]) and v["depmap"] > -0.2]
        print("   non-essential residual median", round(float(np.median(rs)), 3) if rs else None,
              "rel-cells IQR", np.round(np.percentile([recs[t]["rel_cells"] for t in recs if not t.startswith("NTC") and np.isfinite(recs[t]["depmap"]) and recs[t]["depmap"] > -0.2], [25, 75]), 2), flush=True)
    json.dump(res, open(outf, "w"), indent=1)


if __name__ == "__main__":
    main()
