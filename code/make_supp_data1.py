"""Supplementary Data 1: the escape fraction of every fitted perturbation in the seven cell types and states of Table S2
(the per-perturbation rows behind Table S2 and Fig. 2b; negative-binomial fits of fate_mixture.py and
fate_mixture_more.py, read from their saved outputs; no number is recomputed).
Columns: cell type, screen, targeted gene, DepMap median gene effect (empty if the gene is not in DepMap), number of cells,
remaining target expression (own-target counts over the counts expected at control level), escape fraction e (weight of
the control-level group), knockdown level of the knocked-down group (r_kd, fraction of control), BIC difference
(uniform minus two-group) and whether two groups are preferred (difference > 10).
Usage: make_supp_data1.py <metrics_mixture.json> <metrics_mixture_more.json> <out csv>."""
import csv, json, math, os, sys

M1, M2, out = sys.argv[1:4]
A, B = json.load(open(M1))["rows"], json.load(open(M2))["rows"]
SETS = [("hESC", "genome-scale CRISPRi (ref. 4)", A["hESC"]), ("K562", "genome-scale, day 8 (ref. 1)", A["K562_gw"]),
        ("K562", "essential genes, day 6 (ref. 1)", B["K562ess"]), ("RPE1", "essential genes, day 7 (ref. 1)", B["RPE1"]),
        ("Jurkat", "essential genes, day 7 (ref. 2)", B["Jurkat"]), ("HepG2", "essential genes, day 7 (ref. 2)", B["HepG2"]),
        ("CD4+ T, resting", "genome-scale, donor 1 (ref. 3)", B["CD4D1rest"])]
f4 = lambda x: "" if x is None or (isinstance(x, float) and not math.isfinite(x)) else "%.4f" % x
n = 0
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["cell_type", "screen", "target_gene", "depmap_gene_effect", "n_cells", "remaining_target_expression",
                "escape_fraction_e", "knockdown_level_r_kd", "delta_bic_uniform_minus_two_groups", "two_groups_preferred"])
    for ct, sc, rows in SETS:
        for r in sorted(rows, key=lambda r: r["target"]):
            nb = r["nb"]
            w.writerow([ct, sc, r["target"], f4(r["essentiality"]), r["n"], f4(r["residual"]), f4(nb["e"]), f4(nb["r_kd"]),
                        "%.1f" % nb["dbic"], "yes" if nb["dbic"] > 10 else "no"]); n += 1
print("rows", n, "->", out)
