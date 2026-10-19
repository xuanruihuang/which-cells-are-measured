"""What predicts how fast a cell line eliminates perturbed cells, at matched dependency? (5090; fate paper C7 step 1;
expectations X1-X4 pre-stated in notes/fate_paper_plan.md §3, commit acf2f14). Maffa et al. 2026: 16 cancer lines, one
library, Cas9 knockout, harvested day 6, so the protocol is fixed and line-to-line differences in "depletion per unit
dependency" come from the cells.
Per line: Theil-Sen regression of log2(depletion) on the line's own DepMap gene effect over its knockouts (depletion and
gene effect from the authors' depletion_vs_crispr_table.csv, as findings be). Depletion rises toward 1 as the gene effect
rises toward 0, so the slope is positive; elimination sensitivity S = slope (steeper = more cells lost per unit of
dependency). Sign convention corrected 10-18 after the first run, where S was defined as -slope.
Candidate explanations, all from the authors' own annotations (no hand-picked gene sets except the proliferation index):
  TP53_damaging        p53_cell_line_properties.csv
  proliferation        mean log2 CP10K of MKI67, TOP2A, CCNB1, CDK1 in the olfactory-control constructs' pseudobulk
  depth / cells        cell_quality_covariates.csv (technical confounders)
  OncotreeLineage      metadata/cell_line_metadata.csv (negative control)
Reports S per line, Mann-Whitney (one-sided) of S between TP53 wild-type and damaged lines with the rank-biserial
effect size, Spearman of S with each covariate, and the partial Spearman of S with TP53 status given proliferation.
Usage: fate_priming_maffa.py <processed dir> <metadata dir> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import theilslopes, spearmanr, mannwhitneyu

PROLIF = ["MKI67", "TOP2A", "CCNB1", "CDK1"]
ORS = ["OR10H2", "OR11H4", "OR13C4", "OR13H1", "OR4D2", "OR9Q1"]
MIN_KO = 20


def partial_spearman(x, y, z):
    """Spearman of x and y with z regressed out (on ranks)."""
    r = lambda v: pd.Series(v).rank().values
    rx, ry, rz = r(x), r(y), r(z)
    res = lambda v: v - np.polyval(np.polyfit(rz, v, 1), rz)
    return float(spearmanr(res(rx), res(ry))[0])


def main():
    d, meta, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
    dep = pd.read_csv(d + "/depletion_vs_crispr_table.csv")
    qual = pd.read_csv(d + "/cell_quality_covariates.csv").set_index("cell_line")
    p53 = pd.read_csv(d + "/p53_cell_line_properties.csv").drop_duplicates("cell_line").set_index("cell_line")
    lin = pd.read_csv(meta + "/cell_line_metadata.csv").set_index("cell_line")
    hdr = pd.read_csv(d + "/pseudobulk_sum_by_grna.csv", nrows=2, header=None)
    lines, grnas = hdr.iloc[0, 1:].astype(str).values, hdr.iloc[1, 1:].astype(str).values
    X = pd.read_csv(d + "/pseudobulk_sum_by_grna.csv", skiprows=2, header=None, index_col=0)
    genes = X.index.astype(str).values; V = np.nan_to_num(X.values.astype(np.float64))
    gpos = {g: j for j, g in enumerate(genes)}
    gene_of = np.array([g.split("_v")[0] for g in grnas])
    rows = []
    for L in sorted(set(lines)):
        s = dep[(dep.cell_line == L) & np.isfinite(dep.gene_effect) & (dep.depletion > 0)]
        s = s[s.target_class != "Olfactory Control"]
        if len(s) < MIN_KO:
            continue
        sl, ic, lo, hi = theilslopes(np.log2(s.depletion.values), s.gene_effect.values)
        cols = [k for k in np.where(lines == L)[0] if gene_of[k] in ORS]
        ctrl = V[:, cols].sum(1); cp = ctrl / ctrl.sum() * 1e6
        prol = float(np.mean([np.log2(cp[gpos[g]] + 1) for g in PROLIF if g in gpos]))
        rows.append(dict(line=L, S=float(sl), S_ci=[float(lo), float(hi)], n_ko=int(len(s)),
                         prolif=prol, tp53_damaging=float(p53.TP53_damaging.get(L, np.nan)),
                         umis=float(qual.avg_umis_per_ko.get(L, np.nan)), cells=float(qual.n_cells_per_ko.get(L, np.nan)),
                         total_umis=float(qual.total_umis_per_ko.get(L, np.nan)), lineage=str(lin.OncotreeLineage.get(L, "?"))))
    R = pd.DataFrame(rows)
    S = R.S.values; wt = R.tp53_damaging.values == 0
    u, p = mannwhitneyu(S[wt], S[~wt], alternative="greater")
    rbc = 2 * u / (wt.sum() * (~wt).sum()) - 1
    res = dict(rules=__doc__, n_lines=len(R), per_line=rows,
               X1=dict(n_wt=int(wt.sum()), n_mut=int((~wt).sum()), median_S_wt=float(np.median(S[wt])),
                       median_S_mut=float(np.median(S[~wt])), mannwhitney_u=float(u), p_one_sided=float(p),
                       rank_biserial=float(rbc)),
               X2=dict(spearman_S_prolif=float(spearmanr(S, R.prolif.values)[0])),
               X3=dict(spearman_S_umis=float(spearmanr(S, R.umis.values)[0]),
                       spearman_S_cells=float(spearmanr(S, R.cells.values)[0]),
                       spearman_S_total_umis=float(spearmanr(S, R.total_umis.values)[0]),
                       lineage_groups={k: [round(float(v), 3) for v in g] for k, g in R.groupby("lineage").S}),
               X4=dict(partial_spearman_S_tp53_given_prolif=partial_spearman(S, (~wt).astype(float), R.prolif.values),
                       spearman_S_tp53=float(spearmanr(S, (~wt).astype(float))[0]),
                       spearman_prolif_tp53=float(spearmanr(R.prolif.values, (~wt).astype(float))[0])))
    json.dump(res, open(out + "/metrics_priming_maffa.json", "w"), indent=1)
    print(R.sort_values("S", ascending=False)[["line", "S", "n_ko", "prolif", "tp53_damaging", "lineage"]].to_string(index=False), flush=True)
    for k in ("X1", "X2", "X3", "X4"):
        print(k, json.dumps(res[k]), flush=True)


if __name__ == "__main__":
    main()
