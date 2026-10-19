"""Fate signatures in Maffa et al. 2026 (5090; fate paper C1 / C3; expectations M1-M3 in notes/fate_paper_plan.md §3,
committed before running). 16 cancer lines, 100 genes (6 olfactory-receptor controls), Cas9 knockout, harvested day 6;
two dual-guide constructs per gene (v1, v2), i.e. two independent units.
Inputs (figshare 33273600, processed.zip): pseudobulk_sum_by_grna.csv (genes x [cell line, construct], raw UMI sums),
depletion_vs_crispr_table.csv (cells per line x knockout, line-specific DepMap gene effect and the authors' dependency
class).
Effects: log2 CPM fold change of each construct against the pooled control constructs of the SAME version (v1 vs OR v1,
v2 vs OR v2), so the two units share no control cells; genes with control >= 1 CP10K in that line; target zeroed.
Response = sum over genes of LFC_v1 * LFC_v2 (cross-signal of independent units). Null: each OR gene against the other
five (leave-one-out control). Depletion = cells / median cells of the OR knockouts in that line.
M1: per-line Spearman(depletion, gene effect). M2: per-line median response per dependency class, ratio to
nondependent. M3: per gene, Spearman across lines of gene effect vs log response (genes with gene-effect range >= 0.5).
Usage: fate_maffa.py <processed dir> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ORS = ["OR10H2", "OR11H4", "OR13C4", "OR13H1", "OR4D2", "OR9Q1"]
CLASSES = ["Strong", "Moderate", "Weak", "Nondependent"]
MIN_CP10K = 1.0


def lfc(x, c):
    return np.log2((x / x.sum() * 1e6 + 1) / (c / c.sum() * 1e6 + 1))


def main():
    d, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
    hdr = pd.read_csv(d + "/pseudobulk_sum_by_grna.csv", nrows=2, header=None)
    lines, grnas = hdr.iloc[0, 1:].astype(str).values, hdr.iloc[1, 1:].astype(str).values
    X = pd.read_csv(d + "/pseudobulk_sum_by_grna.csv", skiprows=2, header=None, index_col=0)
    genes = X.index.astype(str).values; X = np.nan_to_num(X.values.astype(np.float64))
    gpos = {g: j for j, g in enumerate(genes)}
    gene_of = np.array([g.split("_v")[0] for g in grnas]); ver = np.array([g.split("_v")[1][0] if "_v" in g else "?" for g in grnas])
    dep = pd.read_csv(d + "/depletion_vs_crispr_table.csv")
    rows, null, m1 = [], [], {}
    for L in sorted(set(lines)):
        cols = {(gene_of[k], ver[k]): k for k in np.where(lines == L)[0]}
        ctrl = {v: X[:, [cols[(o, v)] for o in ORS if (o, v) in cols]].sum(1) for v in ("1", "2")}
        call = ctrl["1"] + ctrl["2"]
        keep = np.where(call / call.sum() * 1e4 >= MIN_CP10K)[0]
        dl = dep[dep.cell_line == L].set_index("assigned_ko")
        orc = dl.loc[[o for o in ORS if o in dl.index], "number_cells"]
        base = float(np.median(orc)) if len(orc) else np.nan

        def resp(g, cA, cB):
            if (g, "1") not in cols or (g, "2") not in cols:
                return np.nan
            a = lfc(X[keep, cols[(g, "1")]], cA[keep]); b = lfc(X[keep, cols[(g, "2")]], cB[keep])
            if g in gpos and gpos[g] in set(keep):
                q = int(np.where(keep == gpos[g])[0][0]); a[q] = 0; b[q] = 0
            return float(a @ b)

        for o in ORS:
            others = [p for p in ORS if p != o]
            cA = X[:, [cols[(p, "1")] for p in others if (p, "1") in cols]].sum(1)
            cB = X[:, [cols[(p, "2")] for p in others if (p, "2") in cols]].sum(1)
            null.append(dict(line=L, gene=o, response=resp(o, cA, cB)))
        for g in sorted(set(gene_of[lines == L]) - set(ORS)):
            if g not in dl.index:
                continue
            r = dl.loc[g]
            rows.append(dict(line=L, gene=g, response=resp(g, ctrl["1"], ctrl["2"]), cells=int(r.number_cells),
                             depletion=float(r.number_cells / base), gene_effect=float(r.gene_effect),
                             dep_class=str(r.dependency_strength_class), n_genes=int(len(keep))))
        sub = [x for x in rows if x["line"] == L]
        m1[L] = float(spearmanr([x["depletion"] for x in sub], [x["gene_effect"] for x in sub])[0])
    R = pd.DataFrame(rows); N = pd.DataFrame(null)
    res = {"rules": __doc__, "M1_spearman_depletion_effect": m1, "M1_median": float(np.median(list(m1.values())))}
    per_line = {}
    for L, g in R.groupby("line"):
        med = {c: float(g[g.dep_class == c].response.median()) for c in CLASSES if (g.dep_class == c).sum() >= 3}
        nn = N[N.line == L].response
        per_line[L] = dict(class_median=med, n_class={c: int((g.dep_class == c).sum()) for c in CLASSES},
                           null_median=float(nn.median()), null_max=float(nn.max()))
    res["per_line"] = per_line
    rat = lambda a, b: [v["class_median"][a] / v["class_median"][b] for v in per_line.values()
                        if a in v["class_median"] and b in v["class_median"] and v["class_median"][b] > 0]
    res["M2"] = {"strong_over_nondep_median": float(np.median(rat("Strong", "Nondependent"))),
                 "strong_over_moderate_median": float(np.median(rat("Strong", "Moderate"))),
                 "moderate_over_nondep_median": float(np.median(rat("Moderate", "Nondependent"))),
                 "weak_over_nondep_median": float(np.median(rat("Weak", "Nondependent"))),
                 "n_lines": {k: len(rat(*k.split("/"))) for k in ("Strong/Nondependent", "Strong/Moderate")},
                 "pooled_class_median": {c: float(R[R.dep_class == c].response.median()) for c in CLASSES},
                 "null_pooled_median": float(N.response.median())}
    # depletion bins (across lines): response vs depletion
    R["dbin"] = pd.cut(R.depletion, [0, 0.25, 0.5, 0.75, 1.0, np.inf])
    res["response_by_depletion_bin"] = {str(k): dict(n=int(len(g)), response_median=float(g.response.median()))
                                        for k, g in R.groupby("dbin", observed=True)}
    m3 = {}
    for g, s in R.groupby("gene"):
        s = s[np.isfinite(s.response) & (s.response > 0)]
        if len(s) >= 8 and s.gene_effect.max() - s.gene_effect.min() >= 0.5:
            m3[g] = float(spearmanr(s.gene_effect, np.log(s.response))[0])
    res["M3"] = {"n_genes": len(m3), "median_spearman": float(np.median(list(m3.values()))),
                 "frac_negative": float(np.mean(np.array(list(m3.values())) < 0)), "per_gene": m3}
    res["rows"] = rows; res["null"] = null
    json.dump(res, open(out + "/metrics_fate_maffa.json", "w"), indent=1)
    print("M1", json.dumps(m1), "median", res["M1_median"])
    print("M2", json.dumps({k: v for k, v in res["M2"].items()}))
    print("depletion bins", json.dumps(res["response_by_depletion_bin"]))
    print("M3", res["M3"]["n_genes"], res["M3"]["median_spearman"], res["M3"]["frac_negative"])
    for L, v in per_line.items():
        print(L, json.dumps(v))


if __name__ == "__main__":
    main()
