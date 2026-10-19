"""Fate redesign §6.3 (A): the negative-binomial mixture fit of fate_mixture.py in further contexts (expectations A1-A4
pre-stated, commit 0788b9a). Runs on whichever server holds the per-cell extractions.
Each argument is label=dir, where dir holds cells.npz (tc, own, mu), ctrl.npz (lib, O, cand, lam_cand) and index.json
(targets). Fitting, analysable-target rule, thinned control and BIC rule are those of fate_mixture.py, unchanged.
Also reported: Spearman correlation of e with DepMap essentiality across targets (negative = more essential, more escape).
Usage: fate_mixture_more.py <out_dir> <depmap csv> label=dir [label=dir ...]. Exploratory."""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import fate_mixture as FM


def essentiality(path):
    with open(path) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        rows = [np.array([float(x) if x else np.nan for x in line.rstrip("\n").split(",")[1:]]) for line in f]
    return dict(zip(sym, np.nanmedian(np.array(rows), 0)))


def main():
    out, dep = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    ess = essentiality(dep); rng = np.random.default_rng(97); res = {"rules": __doc__}; allrows = {}
    for arg in sys.argv[3:]:
        lab, d = arg.split("=", 1)
        idx = json.load(open(d + "/index.json")); ctrl = np.load(d + "/ctrl.npz"); cells = np.load(d + "/cells.npz")
        rows = FM.run(lab, cells, ctrl, ctrl["lam_cand"], ctrl["cand"], idx["targets"], ess, rng)
        allrows[lab] = rows
        s = FM.summarize(rows)
        E = np.array([r["essentiality"] for r in rows]); e = np.array([r["nb"]["e"] for r in rows])
        ok = np.isfinite(E) & np.isfinite(e)
        s["spearman_e_vs_depmap"] = float(spearmanr(E[ok], e[ok])[0]) if ok.sum() > 10 else None
        s["n_analysable"] = len(rows)
        res[lab] = s
        print(lab, json.dumps(s, default=float), flush=True)
    res["rows"] = allrows
    json.dump(res, open(out + "/metrics_mixture_more.json", "w"), indent=1, default=lambda v: None if v is None else float(v))


if __name__ == "__main__":
    main()
