"""C7 step 2 (fate paper; expectations J0-J4 pre-stated in notes/fate_paper_plan.md, commit 1a29a8f): within one context,
does the cost of a perturbation set how far its strongest guide's survivors are enriched for escapers, and how far its
response is diluted? Jost et al. 2020 (GSE132080), K562 CRISPRi, 25 genes x ~5 mismatched guides of graded activity.
Per guide g (cells with good coverage and a single guide):
  residual r_g = target raw count / (non-targeting rate x library)
  effect E_g  = log2((raw sum + 0.5) / library) - log2(control rate); response genes: control >= 0.5 CP10K; target zeroed
  p_g = E_g . E_ref / sqrt(M_ref): projection of g on the gene's full response direction, independent units (different
        guides are different transductions); ref = the gene's highest-activity guide (relative_activity_day10), and for
        the ref itself the second-highest; M_ref = random-half (crc32) cross-signal of the ref guide
  dilution D = p(top guide) / max_g p_g ; cost = gamma_day10 of the top guide
Sensitivity: p_g against the leave-one-out mean direction of the gene's other guides.
J0: bootstrap over cells within guides (B = 500): per-gene SE of D and r_top, Cochran Q across genes.
Usage: fate_jost.py <jost dir> <out json>. Exploratory."""
import os, sys, json, gzip, zlib
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import spearmanr, chi2

B = 500
MIN_CP10K = 0.5


def main():
    d, outf = sys.argv[1:3]
    ph = pd.read_csv(d + "/GSE132080_sgRNA_barcode_sequences_and_phenotypes.csv.gz")
    ph = ph.set_index("sgRNA_name")
    ci = pd.read_csv(d + "/GSE132080_cell_identities.csv.gz")
    ci = ci[(ci.good_coverage == True) & (ci.number_of_cells == 1)]
    bcs = pd.read_csv(d + "/GSE132080_10X_barcodes.tsv.gz", header=None)[0].astype(str).values
    gn = pd.read_csv(d + "/GSE132080_10X_genes.tsv.gz", header=None, sep="\t")
    sym = gn[1].astype(str).values
    print("reading matrix", flush=True)
    cache = os.path.join(os.path.dirname(outf), "jost_matrix_cells_x_genes.npz")
    if os.path.exists(cache):
        from scipy.sparse import load_npz
        X = load_npz(cache).tocsr()
    else:
        from scipy.sparse import save_npz
        X = mmread(gzip.open(d + "/GSE132080_10X_matrix.mtx.gz")).tocsc().T.tocsr()   # cells x genes
        save_npz(cache, X)
    print("matrix", X.shape, flush=True)
    bpos = {b: i for i, b in enumerate(bcs)}
    ci = ci[ci.cell_barcode.isin(bpos)]
    rows = np.array([bpos[b] for b in ci.cell_barcode]); gid = ci.guide_identity.astype(str).values
    lib = np.asarray(X.sum(1)).ravel()
    isc = np.array([g.startswith("neg_ctrl") for g in gid])
    def sg(g):
        return g.split("_", 1)[1] if ("_" in g and not g.startswith("neg_ctrl")) else g
    sgn = np.array([sg(g) for g in gid])
    ctrl_rows = rows[isc]
    csum = np.asarray(X[ctrl_rows].sum(0)).ravel(); clib = lib[ctrl_rows].sum()
    resp = np.where(csum / clib * 1e4 >= MIN_CP10K)[0]
    rate = csum / clib; lpc = np.log2((csum[resp] + 0.5) / clib)
    half = np.array([zlib.crc32(b.encode()) & 1 for b in ci.cell_barcode.values])
    print("cells", len(rows), "controls", isc.sum(), "response genes", len(resp), flush=True)
    gpos = {s: j for j, s in enumerate(sym)}
    genes = sorted(set(ph.gene))
    rng = np.random.default_rng(5)
    res = {"rules": __doc__, "genes": {}}
    for G in genes:
        guides = [s for s in ph.index if ph.gene[s] == G and (sgn == s).any()]
        if len(guides) < 3 or G not in gpos:
            print("skip", G, len(guides), G in gpos, flush=True); continue
        tj = gpos[G]; tsel = np.where(resp == tj)[0]
        act = {s: float(ph.relative_activity_day10[s]) for s in guides}
        order = sorted(guides, key=lambda s: -act[s]); top, second = order[0], order[1]
        cells = {s: rows[sgn == s] for s in guides}; hv = {s: half[sgn == s] for s in guides}
        Xg = {s: X[cells[s]][:, resp].toarray().astype(np.float64) for s in guides}
        own = {s: np.asarray(X[cells[s]][:, tj].todense()).ravel() for s in guides}

        def stats(w):
            """w: per-guide cell weights (bootstrap) -> dict of r, p, D, r_top."""
            E, r, Mh = {}, {}, {}
            for s in guides:
                ws = w[s]; L = (ws * lib[cells[s]]).sum()
                e = np.log2(((ws[:, None] * Xg[s]).sum(0) + 0.5) / L) - lpc
                if len(tsel): e[tsel] = 0
                E[s] = e; r[s] = float((ws * own[s]).sum() / (rate[tj] * L)) if rate[tj] > 0 else np.nan
                hs = []
                for h in (0, 1):
                    m = hv[s] == h; Lh = (ws[m] * lib[cells[s]][m]).sum()
                    eh = np.log2(((ws[m][:, None] * Xg[s][m]).sum(0) + 0.5) / max(Lh, 1)) - lpc
                    if len(tsel): eh[tsel] = 0
                    hs.append(eh)
                Mh[s] = float(hs[0] @ hs[1])
            p = {}
            for s in guides:
                ref = second if s == top else top
                p[s] = float(E[s] @ E[ref] / np.sqrt(Mh[ref])) if Mh[ref] > 0 else np.nan
            p_loo = {}
            for s in guides:
                o = np.mean([E[u] for u in guides if u != s], 0)
                p_loo[s] = float(E[s] @ o / np.linalg.norm(o))
            mx = np.nanmax(list(p.values())); mxl = np.nanmax(list(p_loo.values()))
            return dict(r=r, p=p, p_loo=p_loo, D=p[top] / mx if mx > 0 else np.nan,
                        D_loo=p_loo[top] / mxl if mxl > 0 else np.nan, r_top=r[top])

        base = stats({s: np.ones(len(cells[s])) for s in guides})
        bs = [stats({s: rng.poisson(1.0, len(cells[s])).astype(float) for s in guides}) for _ in range(B)]
        se = lambda k: float(np.nanstd([b[k] for b in bs]))
        rho_r_act = float(spearmanr([act[s] for s in guides], [base["r"][s] for s in guides])[0])
        res["genes"][G] = dict(n_guides=len(guides), cells={s: int(len(cells[s])) for s in guides}, activity=act,
                               gamma_top=float(ph.gamma_day10[top]), top=top, r=base["r"], p=base["p"], p_loo=base["p_loo"],
                               D=base["D"], D_se=se("D"), D_loo=base["D_loo"], D_loo_se=se("D_loo"),
                               r_top=base["r_top"], r_top_se=se("r_top"), spearman_r_activity=rho_r_act)
        g = res["genes"][G]
        print("%-8s gamma_top %.3f  D %.2f (%.2f)  D_loo %.2f  r_top %.2f (%.2f)  rho(r,act) %.2f  cells %s" % (
            G, g["gamma_top"], g["D"], g["D_se"], g["D_loo"], g["r_top"], g["r_top_se"], rho_r_act,
            [g["cells"][s] for s in sorted(guides, key=lambda s: -act[s])]), flush=True)
    R = res["genes"]; Gs = list(R)
    def Q(k):
        v = np.array([R[g][k] for g in Gs]); s = np.array([R[g][k + "_se"] for g in Gs])
        ok = np.isfinite(v) & np.isfinite(s) & (s > 0); v, s = v[ok], s[ok]; w = 1 / s ** 2
        q = float((w * (v - (w * v).sum() / w.sum()) ** 2).sum()); return dict(Q=q, df=int(ok.sum() - 1), p=float(chi2.sf(q, ok.sum() - 1)))
    gam = np.array([R[g]["gamma_top"] for g in Gs])
    summ = dict(J0_D=Q("D"), J0_r_top=Q("r_top"), J0_D_loo=Q("D_loo"),
                J1_median_spearman_r_activity=float(np.nanmedian([R[g]["spearman_r_activity"] for g in Gs])),
                J2_spearman_gamma_D=float(spearmanr(gam, [R[g]["D"] for g in Gs], nan_policy="omit")[0]),
                J2_spearman_gamma_D_loo=float(spearmanr(gam, [R[g]["D_loo"] for g in Gs], nan_policy="omit")[0]),
                J3_spearman_gamma_rtop=float(spearmanr(gam, [R[g]["r_top"] for g in Gs], nan_policy="omit")[0]),
                J4_median_D_weak=float(np.nanmedian([R[g]["D"] for g in Gs if R[g]["gamma_top"] > -0.15])),
                J4_n_weak=int(sum(R[g]["gamma_top"] > -0.15 for g in Gs)), n_genes=len(Gs))
    res["summary"] = summ
    print(json.dumps(summ, indent=1), flush=True)
    json.dump(res, open(outf, "w"), indent=1)


if __name__ == "__main__":
    main()
