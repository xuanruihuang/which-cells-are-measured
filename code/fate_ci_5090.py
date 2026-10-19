"""Intervals for the fate paper, 5090 part (2026-10-18; for notes/fate_paper_ledger.md; same conventions as
code/a100/fate_ci.py): Jurkat / HepG2 budget (target bootstrap) and Maffa 16 lines (lines are independent contexts:
spread over lines plus a bootstrap over lines; M3 bootstrap over genes).
Usage: fate_ci_5090.py <budget metrics json> <maffa metrics json> <out json>. Exploratory."""
import sys, json
import numpy as np

B = 2000
TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf))
rng = np.random.default_rng(2026)


def share(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float); ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
    return float(np.median(num[ok] / den[ok])) if ok.sum() else np.nan


def boot(fn, *arrs):
    arrs = [np.asarray(a, float) for a in arrs]; n = len(arrs[0]); est = fn(*arrs)
    if n < 5:
        return dict(est=est, ci=None, n=n)
    bs = np.array([fn(*[a[i] for a in arrs]) for i in (rng.integers(0, n, n) for _ in range(B))], float)
    bs = bs[np.isfinite(bs)]
    return dict(est=est, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n=n)


def main():
    bj, mj, outf = sys.argv[1:4]; out = {"rules": __doc__, "bm": {}}
    M = json.load(open(bj))
    for lab in ("Jurkat", "HepG2"):
        R = M[lab]["rows_random"]; d = {}
        for nm, lo, hi in TIERS:
            ts = [t for t, r in R.items() if np.isfinite(r["essentiality"]) and lo <= r["essentiality"] < hi]
            mt = [R[t]["m_total"] for t in ts]
            d[nm] = dict(comp_share=boot(share, [R[t]["m_comp"] for t in ts], mt),
                         selection=boot(share, [R[t].get("m_kd", np.nan) for t in ts], mt))
        out["bm"][lab] = d
    F = json.load(open(mj)); pl = F["per_line"]
    rat = [v["class_median"]["Strong"] / v["class_median"]["Nondependent"] for v in pl.values()
           if "Strong" in v["class_median"] and "Nondependent" in v["class_median"] and v["class_median"]["Nondependent"] > 0]
    m3 = list(F["M3"]["per_gene"].values())
    out["be"] = dict(M2_strong_over_nondep_per_line=sorted(round(x, 2) for x in rat),
                     M2_median_over_lines=boot(lambda v: float(np.median(v)), rat),
                     M2_iqr_over_lines=[float(np.percentile(rat, 25)), float(np.percentile(rat, 75))],
                     M3_median_over_genes=boot(lambda v: float(np.median(v)), m3),
                     M3_frac_negative=boot(lambda v: float(np.mean(v < 0)), m3))
    json.dump(out, open(outf, "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
