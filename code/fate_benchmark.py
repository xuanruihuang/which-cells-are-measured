"""Does fate correction change what a benchmark concludes? (fate paper C5; expectations W1-W4 pre-stated in
notes/fate_paper_plan.md §3, commit ef08935). Input: fate_vectors_export.py outputs.
A source context's response is used as the prediction for a target context, on shared targets and the intersection of
their response genes. Ground truth = the target context's effect, under three definitions ("fate corrections applied to
the benchmark data itself"): total, within-cell (total minus composition), knocked-down cells only (thinning null
subtracted). Each ground truth is the mean of that context's two unit sides.
Predictors: raw = the source context's total effect; scaled = raw times a per-tier scalar (sqrt of the paired median
magnitude ratio truth/raw, so only the magnitude changes); zero = all zeros (reference floor).
Metrics per target, median per DepMap tier: cosine similarity; relative MSE = ||pred - truth||^2 / ||truth||^2 (1.0 is
the zero predictor); discrimination = corr(pred, own truth) minus the mean corr(pred, other targets' truths) (the
cell-eval perturbation-discrimination analogue, computed within the tier).
Each tier is reported twice: over every target where that ground truth exists, and over the common subset of targets
where all three definitions exist (the groupable, highly expressed targets), so that differences between truth
definitions are not confounded by the target subset.
Usage: fate_benchmark.py <depmap csv> <vectors dir> <source label> <target label> [<target label> ...]. Exploratory."""
import sys, json
import numpy as np

TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf))
GUIDE_CONTEXTS = {"hESC", "CD4_D4rest", "CD4_D2rest", "CD4_D3rest"}
MIN_TIER = 10


def essentiality(path):
    with open(path) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        rows = [np.array([float(x) if x else np.nan for x in line.rstrip("\n").split(",")[1:]]) for line in f]
    return dict(zip(sym, np.nanmedian(np.array(rows), 0)))


def load(d, lab):
    z = np.load("%s/%s_vectors.npz" % (d, lab), allow_pickle=True)
    return dict(lab=lab, targets=[str(t) for t in z["targets"]], genes=[str(g) for g in z["genes"]],
                **{k: z[k] for k in ("tot_r", "comp_r", "kd_r", "kdcomp_r", "tot_g", "comp_g")})


def mean_vector(C, k, kind):
    """Mean of the two unit sides for target index k under one ground-truth / predictor definition, or None."""
    use_g = C["lab"] in GUIDE_CONTEXTS and np.isfinite(C["tot_g"][k, 0, 0])
    if kind in ("total", "within"):
        t, c = (C["tot_g"], C["comp_g"]) if use_g else (C["tot_r"], C["comp_r"])
        v = t[k] if kind == "total" else t[k] - c[k]
    else:
        v = C["kd_r"][k] if kind == "kd" else C["kd_r"][k] - C["kdcomp_r"][k]
    return None if not np.isfinite(v[0, 0]) else v.mean(0)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def main():
    dep, d, src = sys.argv[1:4]; ess = essentiality(dep)
    S = load(d, src); res = {"rules": __doc__, "source": src}
    for tlab in sys.argv[4:]:
        X = load(d, tlab)
        gS = {g: j for j, g in enumerate(S["genes"])}
        shared_g = [g for g in X["genes"] if g in gS]
        ix = np.array([X["genes"].index(g) for g in shared_g]); isr = np.array([gS[g] for g in shared_g])
        tS = {t: k for k, t in enumerate(S["targets"])}
        out = {"n_shared_genes": len(shared_g)}
        common = set()
        for kx, t in enumerate(X["targets"]):
            if t in tS and all(mean_vector(X, kx, k) is not None for k in ("total", "within", "kd")) \
                    and mean_vector(S, tS[t], "total") is not None:
                common.add(t)
        out["n_common_subset"] = len(common)
        for truth_kind in ("total", "within", "kd", "total_common", "within_common", "kd_common"):
            base = truth_kind.replace("_common", ""); only_common = truth_kind.endswith("_common")
            per_tier = {}
            for nm, lo, hi in TIERS:
                recs = []
                for kx, t in enumerate(X["targets"]):
                    e = ess.get(t, np.nan)
                    if t not in tS or not np.isfinite(e) or not (lo <= e < hi):
                        continue
                    if only_common and t not in common:
                        continue
                    tv = mean_vector(X, kx, base); pv = mean_vector(S, tS[t], "total")
                    if tv is None or pv is None:
                        continue
                    recs.append((t, tv[ix], pv[isr]))
                if len(recs) < MIN_TIER:
                    per_tier[nm] = dict(n=len(recs)); continue
                T = np.array([r[1] for r in recs]); P = np.array([r[2] for r in recs])
                nt = (T ** 2).sum(1)
                mr = np.median(np.sqrt(np.maximum(nt, 1e-12)) / np.maximum(np.sqrt((P ** 2).sum(1)), 1e-12))
                sc = {"raw": P, "scaled": P * mr, "zero": np.zeros_like(P)}
                row = dict(n=len(recs), scale_factor=float(mr))
                for pname, PR in sc.items():
                    rel = ((PR - T) ** 2).sum(1) / np.maximum(nt, 1e-12)
                    cos = np.array([cosine(PR[i], T[i]) for i in range(len(recs))])
                    # discrimination: own-truth correlation minus mean correlation with the other truths
                    Tc = T - T.mean(1, keepdims=True); Pc = PR - PR.mean(1, keepdims=True)
                    Tn = Tc / np.maximum(np.linalg.norm(Tc, axis=1, keepdims=True), 1e-12)
                    Pn = Pc / np.maximum(np.linalg.norm(Pc, axis=1, keepdims=True), 1e-12)
                    Cm = Pn @ Tn.T
                    own = np.diag(Cm); oth = (Cm.sum(1) - own) / max(len(recs) - 1, 1)
                    row[pname] = dict(rel_mse=float(np.median(rel)), cosine=float(np.nanmedian(cos)),
                                      discrimination=float(np.median(own - oth)))
                per_tier[nm] = row
            out[truth_kind] = per_tier
        res[tlab] = out
        print("==", tlab, "truth <-", src, "shared genes", len(shared_g), "common subset", len(common), flush=True)
        for truth_kind in ("total", "within", "kd", "total_common", "within_common", "kd_common"):
            for nm, v in out[truth_kind].items():
                if v.get("n", 0) < MIN_TIER:
                    continue
                print("  truth=%-14s %-10s n=%4d scale=%.2f | raw mse %.2f cos %.3f disc %.3f | scaled mse %.2f cos %.3f disc %.3f | zero mse %.2f" % (
                    truth_kind, nm, v["n"], v["scale_factor"], v["raw"]["rel_mse"], v["raw"]["cosine"], v["raw"]["discrimination"],
                    v["scaled"]["rel_mse"], v["scaled"]["cosine"], v["scaled"]["discrimination"], v["zero"]["rel_mse"]), flush=True)
    json.dump(res, open("%s/metrics_benchmark_%s.json" % (d, src), "w"), indent=1)


if __name__ == "__main__":
    main()
