"""Method audit of findings (bq), fate paper C7 step 1 (2026-10-18). Read-only on the Maffa 2026 processed tables.
Question: fate_priming_maffa.py drops knockouts with depletion == 0 and regresses log2(depletion) on gene effect. In a
line with few cells per knockout, the most depleted knockouts hit zero (dropped) or sit near the counting floor
(1 / median control cells), which flattens the slope. That alone would make S correlate with cells per knockout
(observed 0.43), and would contaminate the X1 / X2 tests.
Per line: number and share of knockouts with depletion == 0; smallest non-zero depletion; S as in (bq); S_floor with the
zeros kept at half the smallest non-zero depletion; S_mid on knockouts with gene effect in [-1, 0] only (above the floor);
then Spearman of each S with cells per knockout, the proliferation index, TP53 status.
Usage: audit_bq_floor.py <processed dir> <metadata dir>. Exploratory audit."""
import sys, json
import numpy as np
import pandas as pd
from scipy.stats import theilslopes, spearmanr

ORS_CLASS = "Olfactory Control"


def main():
    d, meta = sys.argv[1:3]
    dep = pd.read_csv(d + "/depletion_vs_crispr_table.csv")
    qual = pd.read_csv(d + "/cell_quality_covariates.csv").set_index("cell_line")
    p53 = pd.read_csv(d + "/p53_cell_line_properties.csv").drop_duplicates("cell_line").set_index("cell_line")
    print("columns", list(dep.columns), flush=True)
    rows = []
    for L, s in dep.groupby("cell_line"):
        s = s[np.isfinite(s.gene_effect) & (s.target_class != ORS_CLASS)]
        if len(s) < 20:
            continue
        z = s.depletion.values; ge = s.gene_effect.values
        nz = z > 0
        floor = float(z[nz].min()) if nz.any() else np.nan
        S0 = theilslopes(np.log2(z[nz]), ge[nz])[0]
        zf = np.where(nz, z, floor / 2)
        Sf = theilslopes(np.log2(zf), ge)[0]
        mid = nz & (ge >= -1) & (ge <= 0)
        Sm = theilslopes(np.log2(z[mid]), ge[mid])[0] if mid.sum() >= 15 else np.nan
        rows.append(dict(line=L, n=len(s), n_zero=int((~nz).sum()), floor=floor,
                         n_le_floor2=int((nz & (z <= 2 * floor)).sum()), S=float(S0), S_floor=float(Sf), S_mid=float(Sm),
                         n_mid=int(mid.sum()), cells=float(qual.n_cells_per_ko.get(L, np.nan)),
                         tp53=float(p53.TP53_damaging.get(L, np.nan))))
    R = pd.DataFrame(rows)
    print(R.to_string(index=False), flush=True)
    out = {}
    for k in ("S", "S_floor", "S_mid"):
        ok = np.isfinite(R[k].values)
        out[k] = dict(spearman_cells=float(spearmanr(R[k][ok], R.cells[ok])[0]),
                      spearman_tp53=float(spearmanr(R[k][ok], R.tp53[ok])[0]), range=[float(R[k].min()), float(R[k].max())])
    out["spearman_cells_nzero"] = float(spearmanr(R.cells, R.n_zero)[0])
    out["spearman_cells_floor"] = float(spearmanr(R.cells, R.floor)[0])
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
