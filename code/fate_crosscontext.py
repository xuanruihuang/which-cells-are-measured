"""Cross-context consistency of perturbation effects under fate-aware estimators (fate paper C6 validation; expectations
V1-V3 pre-stated in notes/fate_paper_plan.md §3, commit 733bd72). Input: fate_vectors_export.py outputs.
For a reference context R and a context X, on shared targets and the intersection of their response genes:
  E_c = mean of the two unit vectors; M_c = unit-0 . unit-1 (within-context reproducible magnitude)
  rho = (E_X . E_R) / sqrt(M_X M_R)   (disattenuated correlation; contexts are independent cells, so the numerator has
                                       no noise bias; requires M_X, M_R > 0)
  magnitude ratio = M_X / M_R (paired median of log2 ratios)
Estimators: total (E), within (E - Ec), kd (knocked-down cells, thinning null subtracted), kd_within (kd - its Ec).
Units: guide parity where the context is listed in GUIDE_CONTEXTS and the target has guide vectors, else random halves.
Per DepMap tier (median gene effect over lines): median rho per estimator on the targets where all compared estimators
exist, and the paired change.
Usage: fate_crosscontext.py <depmap csv> <vectors dir> <reference label> <label> [<label> ...]. Exploratory."""
import sys, json
import numpy as np

TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf))
GUIDE_CONTEXTS = {"hESC", "CD4_D4rest", "CD4_D2rest", "CD4_D3rest"}


def essentiality(path):
    with open(path) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        rows = [np.array([float(x) if x else np.nan for x in line.rstrip("\n").split(",")[1:]]) for line in f]
    return dict(zip(sym, np.nanmedian(np.array(rows), 0)))


def load(d, lab):
    z = np.load("%s/%s_vectors.npz" % (d, lab), allow_pickle=True)
    return dict(lab=lab, targets=[str(t) for t in z["targets"]], genes=[str(g) for g in z["genes"]],
                **{k: z[k] for k in ("tot_r", "comp_r", "kd_r", "kdcomp_r", "tot_g", "comp_g")})


def units(C, k, est):
    """Two unit vectors for target index k under estimator est, or None."""
    use_g = C["lab"] in GUIDE_CONTEXTS and np.isfinite(C["tot_g"][k, 0, 0])
    if est in ("total", "within"):
        t, c = (C["tot_g"], C["comp_g"]) if use_g else (C["tot_r"], C["comp_r"])
        v = t[k] if est == "total" else t[k] - c[k]
    else:
        t, c = C["kd_r"], C["kdcomp_r"]
        v = t[k] if est == "kd" else t[k] - c[k]
    return None if not np.isfinite(v[0, 0]) else v


def compare(R, X, ess, ests):
    gR = {g: j for j, g in enumerate(R["genes"])}; shared = [g for g in X["genes"] if g in gR]
    ix = np.array([X["genes"].index(g) for g in shared]); ir = np.array([gR[g] for g in shared])
    tR = {t: k for k, t in enumerate(R["targets"])}
    rows = {}
    for kx, t in enumerate(X["targets"]):
        if t not in tR:
            continue
        kr = tR[t]; r = {"essentiality": float(ess.get(t, np.nan))}
        for est in ests:
            vx, vr = units(X, kx, est), units(R, kr, est)
            if vx is None or vr is None:
                continue
            ux, ur = vx[:, ix], vr[:, ir]
            Mx, Mr = float(ux[0] @ ux[1]), float(ur[0] @ ur[1])
            cross = float(ux.mean(0) @ ur.mean(0))
            r[est] = dict(rho=cross / np.sqrt(Mx * Mr) if Mx > 0 and Mr > 0 else np.nan, Mx=Mx, Mr=Mr)
        rows[t] = r
    return rows, len(shared)


def summarize(rows, a, b):
    out = {}
    for nm, lo, hi in TIERS:
        ts = [t for t, r in rows.items() if np.isfinite(r["essentiality"]) and lo <= r["essentiality"] < hi
              and a in r and b in r and np.isfinite(r[a]["rho"]) and np.isfinite(r[b]["rho"])]
        if not ts:
            out[nm] = dict(n=0); continue
        ra = np.array([rows[t][a]["rho"] for t in ts]); rb = np.array([rows[t][b]["rho"] for t in ts])
        ma = np.array([rows[t][a]["Mx"] / rows[t][a]["Mr"] for t in ts]); mb = np.array([rows[t][b]["Mx"] / rows[t][b]["Mr"] for t in ts])
        pos = (ma > 0) & (mb > 0)
        out[nm] = dict(n=len(ts), rho_a=float(np.median(ra)), rho_b=float(np.median(rb)),
                       delta_rho_median=float(np.median(rb - ra)), frac_improved=float(np.mean(rb > ra)),
                       mag_ratio_a=float(2 ** np.median(np.log2(ma[pos]))) if pos.any() else None,
                       mag_ratio_b=float(2 ** np.median(np.log2(mb[pos]))) if pos.any() else None)
    return out


def main():
    dep, d, ref = sys.argv[1:4]; ess = essentiality(dep)
    R = load(d, ref); res = {"rules": __doc__, "reference": ref}
    for lab in sys.argv[4:]:
        X = load(d, lab)
        rows, ng = compare(R, X, ess, ("total", "within", "kd", "kd_within"))
        res[lab] = {"n_shared_genes": ng, "n_shared_targets": len(rows),
                    "total_vs_within": summarize(rows, "total", "within"),
                    "total_vs_kd": summarize(rows, "total", "kd"),
                    "total_vs_kd_within": summarize(rows, "total", "kd_within")}
        print("==", lab, "vs", ref, "shared genes", ng, "shared targets", len(rows), flush=True)
        for comp in ("total_vs_within", "total_vs_kd", "total_vs_kd_within"):
            for nm, v in res[lab][comp].items():
                print("  %-18s %-10s %s" % (comp, nm, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})), flush=True)
    json.dump(res, open("%s/metrics_crosscontext_%s.json" % (d, ref), "w"), indent=1)


if __name__ == "__main__":
    main()
