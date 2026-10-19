"""Fate signatures in the Virtual Cell Challenge 2025 H1 hESC data (5090; fate paper C1 / C5). The VCC used these cells
as the test context for perturbation prediction, so if strongly perturbed H1 cells are eliminated (as in Pan's hESC,
findings m, ba, bb), the "ground-truth" responses of essential targets are survivor-diluted.
Data: official adata_{Training,Validation,Test}.h5ad (raw counts, obs target_gene / guide_id / batch). One element per
gene and one transduction, so no guide halves: responses use random cell halves (crc32 of the barcode) and are flagged
as such; depletion, knockdown residual and dG1 do not need halves.
Per split and target (>= MIN cells): cells; knockdown residual = own-target raw / (control rate x library), control
rate from non-targeting cells of the same split; dG1 = G1 fraction minus control (Tirosh half-1 genes, seed 7, vs 10
expression-matched controls each, as pan_extract.py); response = cross-half signal of pseudobulk log1p CP10K effects
over genes with control mean >= 0.1 (target zeroed). DepMap tiers as fate_iu_cd4.py.
Usage: fate_h1.py <vcc2025 dir> <depmap csv> <out_dir>. Exploratory."""
import os, sys, json, zlib
import numpy as np
import h5py
from scipy.sparse import csr_matrix

MIN = 20
TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf))
S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1",
           "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3",
           "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1",
           "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
G2M = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO",
       "CENPF", "TACC3", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1",
       "KIF20B", "HJURP", "CDCA3", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8",
       "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3",
       "CBX5", "CENPA"]


def essentiality(path):
    with open(path) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        rows = [np.array([float(x) if x else np.nan for x in line.rstrip("\n").split(",")[1:]]) for line in f]
    return dict(zip(sym, np.nanmedian(np.array(rows), 0)))


def cat(o, k):
    n = o[k]
    if isinstance(n, h5py.Group):
        c = np.array([x.decode() if isinstance(x, bytes) else x for x in n["categories"][:]]); return c[n["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in n[:]])


def load(path):
    with h5py.File(path) as f:
        X = f["X"]
        if isinstance(X, h5py.Group):
            M = csr_matrix((X["data"][:], X["indices"][:], X["indptr"][:]), shape=tuple(X.attrs["shape"]))
        else:
            M = csr_matrix(X[:])
        genes = cat(f["var"], "_index"); tg = cat(f["obs"], "target_gene"); bc = cat(f["obs"], "_index")
    return M.astype(np.float64), genes, tg, bc


def run(path, ess):
    M, genes, tg, bc = load(path)
    lib = np.asarray(M.sum(1)).ravel(); gi = {g: j for j, g in enumerate(genes)}
    ctrl = [c for c in ("non-targeting", "non_targeting", "NTC") if c in set(tg)][0]
    Y = M.multiply(1e4 / lib[:, None]).tocsr(); Y.data = np.log1p(Y.data)
    isc = tg == ctrl
    mu = np.asarray(Y[isc].mean(0)).ravel()
    rng = np.random.default_rng(7)
    sS = [g for g in S_GENES if g in gi]; sS = list(rng.permutation(sS))[: len(sS) // 2]
    sG = [g for g in G2M if g in gi]; sG = list(rng.permutation(sG))[: len(sG) // 2]
    iS, iG = np.array([gi[g] for g in sS]), np.array([gi[g] for g in sG])
    bins = np.digitize(mu, np.quantile(mu, np.linspace(0, 1, 26)[1:-1])); r = np.random.default_rng(100)
    cS, cG = (np.concatenate([r.choice(np.where(bins == bins[j])[0], 10, replace=False) for j in ix]) for ix in (iS, iG))
    scS = np.asarray(Y[:, iS].mean(1)).ravel() - np.asarray(Y[:, cS].mean(1)).ravel()
    scG = np.asarray(Y[:, iG].mean(1)).ravel() - np.asarray(Y[:, cG].mean(1)).ravel()
    g1 = (scS <= 0) & (scG <= 0)
    half = np.array([zlib.crc32(b.encode()) & 1 for b in bc])
    gk = np.where(mu >= 0.1)[0]; kpos = {j: q for q, j in enumerate(gk)}
    lam = np.asarray(M[isc].sum(0)).ravel() / lib[isc].sum()
    cA = np.asarray(Y[isc & (half == 0)][:, gk].mean(0)).ravel(); cB = np.asarray(Y[isc & (half == 1)][:, gk].mean(0)).ravel()
    g1c = g1[isc].mean()
    rows = []
    for t in sorted(set(tg) - {ctrl}):
        m = tg == t
        if m.sum() < MIN or t not in gi:
            continue
        j = gi[t]
        res = M[m][:, j].sum() / max(lam[j] * lib[m].sum(), 1e-12)
        a = np.asarray(Y[m & (half == 0)][:, gk].mean(0)).ravel() - cA
        b = np.asarray(Y[m & (half == 1)][:, gk].mean(0)).ravel() - cB
        if j in kpos:
            a[kpos[j]] = 0; b[kpos[j]] = 0
        rows.append(dict(target=t, cells=int(m.sum()), residual=float(res), dG1=float(g1[m].mean() - g1c),
                         response=float(a @ b), control_mean=float(mu[j]), essentiality=float(ess.get(t, np.nan))))
    return rows, dict(n_control=int(isc.sum()), g1_control=float(g1c), umi_per_cell=float(lib.mean()))


def main():
    d, dep, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
    ess = essentiality(dep); res = {"rules": __doc__}
    for split in ("Training", "Validation", "Test"):
        rows, meta = run(os.path.join(d, "adata_%s.h5ad" % split), ess)
        e = np.array([r["essentiality"] for r in rows]); cells = np.array([r["cells"] for r in rows])
        resid = np.array([r["residual"] for r in rows]); dg1 = np.array([r["dG1"] for r in rows]); resp = np.array([r["response"] for r in rows])
        expd = np.array([r["control_mean"] for r in rows]) >= 0.5
        tiers = {}
        for name, lo, hi in TIERS:
            m = np.isfinite(e) & (e >= lo) & (e < hi)
            if not m.any():
                continue
            tiers[name] = {"n": int(m.sum()), "log10_cells_median": float(np.median(np.log10(cells[m]))),
                           "residual_median_expressed": float(np.median(resid[m & expd])) if (m & expd).any() else None,
                           "dG1_median": float(np.median(dg1[m])), "response_random_median": float(np.median(resp[m]))}
        res[split] = {"meta": meta, "n_targets": len(rows), "tiers": tiers,
                      "spearman_cells_essentiality": None, "rows": rows}
        print(split, json.dumps(meta), len(rows), flush=True)
        for k, v in tiers.items():
            print("   ", k, json.dumps(v), flush=True)
    json.dump(res, open(out + "/metrics_fate_h1.json", "w"), indent=1)


if __name__ == "__main__":
    main()
