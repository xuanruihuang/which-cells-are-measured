"""Build the worked example from the K562 genome-wide day-8 screen (Replogle et al. 2022, raw single cells part 01).

Input: the per-cell extraction of code/k562gw_percell_own.py (cells.npz, ctrl.npz, index.json) and the paper's fit
of it (metrics_mixture.json from code/fate_mixture.py, read only to choose targets).
Targets, by a fixed rule: in each DepMap stratum, the analysable targets at the 25th, 50th and 75th percentile of the
paper's e (12 targets), plus the candidate target with the fewest cells (fewer than 30, to show an unfitted row).
Controls: 10,000 non-targeting cells drawn at random (seed 1), one row per cell and target gene.
Library = the cell's UMI count (recovered as mu / control rate for perturbed cells).
Usage: make_example.py <percell_k562gw dir> <metrics_mixture.json> <out_dir>"""
import gzip, json, os, sys
import numpy as np

STRATA = ((-9, -1.0), (-1.0, -0.5), (-0.5, -0.2), (-0.2, 9))


def main():
    d, mix, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
    idx = json.load(open(d + "/index.json")); T = idx["targets"]
    z = np.load(d + "/cells.npz"); c = np.load(d + "/ctrl.npz")
    cand, lam, O, lib = list(c["cand"]), c["lam_cand"], c["O"], c["lib"]
    rows = json.load(open(mix))["rows"]["K562_gw"]
    pick = []
    for lo, hi in STRATA:
        rr = sorted([r for r in rows if r["essentiality"] is not None and lo <= r["essentiality"] < hi],
                    key=lambda r: r["nb"]["e"])
        pick += [rr[int(q * (len(rr) - 1))]["target"] for q in (0.25, 0.5, 0.75)]
    n = np.bincount(z["tc"], minlength=len(T))
    pick.append(T[min(cand, key=lambda t: n[t])])
    ref = {r["target"]: r for r in rows}
    with gzip.open(out + "/cells.csv.gz", "wt", newline="") as f:
        f.write("perturbation,gene,count,library\n")
        for g in pick:
            t = T.index(g); j = cand.index(t); m = z["tc"] == t
            for own, mu in zip(z["own"][m], z["mu"][m]):
                f.write("%s,%s,%d,%d\n" % (g, g, own, round(mu / lam[j])))
    sel = np.sort(np.random.default_rng(1).choice(len(lib), size=10000, replace=False))
    with gzip.open(out + "/controls.csv.gz", "wt", newline="") as f:
        f.write("gene,count,library\n")
        for g in pick:
            j = cand.index(T.index(g))
            for i in sel:
                f.write("%s,%d,%d\n" % (g, O[i, j], round(lib[i])))
    with open(out + "/paper_fit.tsv", "w") as f:
        f.write("perturbation\tn_cells\tessentiality\te\tdbic\n")
        for g in pick:
            r = ref.get(g)
            f.write("%s\t%d\t%s\t%s\t%s\n" % (g, n[T.index(g)], "NA" if r is None else "%.3f" % r["essentiality"],
                                              "NA" if r is None else "%.4g" % r["nb"]["e"],
                                              "NA" if r is None else "%.4g" % r["nb"]["dbic"]))
    print("targets", pick)


if __name__ == "__main__":
    main()
