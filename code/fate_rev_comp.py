"""Fate v3r2 external review, check V10 (expectation pre-stated in notes/fate_analysis_redesign.md §7.2, commit f9d9a81).
Runs on either server. Per target of fate_budget.py rows_random (random halves): c = M_comp / M_total, w = M_within /
M_total, cross term x = 1 - c - w (= 2 <E_c, E_w> / M_total), symmetric (Shapley) composition share (1 + c - w) / 2;
medians over targets with M_total > 0 per DepMap stratum, bootstrap 2,000.
Usage: fate_rev_comp.py <out json> <metrics_fate_budget.json> [...]. Exploratory."""
import sys, json
import numpy as np

TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf))


def bm(x, rng):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return None
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(2000)]
    return [float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    out = sys.argv[1]; rng = np.random.default_rng(5); res = {"rules": __doc__}
    for f in sys.argv[2:]:
        d = json.load(open(f))
        for lab, v in d.items():
            if lab == "rules" or "rows_random" not in v:
                continue
            rows = v["rows_random"]; R = {}
            for nm, lo, hi in TIERS:
                rr = [r for r in rows.values() if r["essentiality"] is not None and np.isfinite(r["essentiality"]) and lo <= r["essentiality"] < hi and r["m_total"] > 0]
                c = np.array([r["m_comp"] / r["m_total"] for r in rr]); w = np.array([r["m_within"] / r["m_total"] for r in rr])
                R[nm] = dict(n=len(rr), comp=bm(c, rng), within=bm(w, rng), cross=bm(1 - c - w, rng), symmetric=bm((1 + c - w) / 2, rng),
                             comp_upper=bm(1 - w, rng))
            res[lab] = R
            print(lab, json.dumps(R["lt_-1"]), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
