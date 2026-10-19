"""R2 (fate paper §8; expectations N1-N4 pre-stated in notes/fate_paper_plan.md §8, commit a42ae7f): does the published
ordering of pairwise agreement among K562, RPE1, Jurkat and HepG2 (Nadig et al.: K562/Jurkat 0.74, RPE1/HepG2 0.75,
K562/HepG2 0.64, Jurkat/HepG2 0.69, RPE1/Jurkat 0.56, K562/RPE1 0.40; "RPE1 especially unique") survive removing the
composition term on both sides, and is it unchanged by matching knockdown or using silenced cells only?
Input: fate_vectors_export.py outputs (<vectors dir>/<label>_vectors.npz; random-half units in all four contexts) and
each context's budget index.json for the per-target survivor residual.
Per pair (A, B), shared targets, intersection of response genes: rho = (E_A . E_B) / sqrt(M_A M_B) with E the mean of
the two unit vectors and M the within-context cross-signal (as fate_crosscontext.py); magnitude ratio M_A / M_B.
Estimators: total; within (E - Ec on both sides); kd (silenced cells, thinning null subtracted, both sides); kd_within.
Target sets: all (rho finite for total and within); fourway (M_total > 0 in all four contexts); matched (residual <= KD
in both). Per DepMap tier and pooled. RPE1 uniqueness = mean rho of the three RPE1 pairs minus the mean of the other
three. Bootstrap over targets (NBOOT) for medians and paired deltas.
Usage: fate_nadig_pairs.py <depmap csv> <vectors dir> <out_dir> <label>:<index.json> [...]. Exploratory."""
import os, sys, json, itertools
import numpy as np

TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf), ("pooled", -np.inf, np.inf))
KD = 0.3
NBOOT = 1000
ESTS = ("total", "within", "kd", "kd_within")


def essentiality(path):
    with open(path) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        rows = [np.array([float(x) if x else np.nan for x in line.rstrip("\n").split(",")[1:]]) for line in f]
    return dict(zip(sym, np.nanmedian(np.array(rows), 0)))


def load(d, lab, index_path):
    z = np.load("%s/%s_vectors.npz" % (d, lab), allow_pickle=True)
    idx = json.load(open(index_path))
    C = dict(lab=lab, targets=[str(t) for t in z["targets"]], genes=[str(g) for g in z["genes"]],
             resid=idx["residual"], **{k: z[k] for k in ("tot_r", "comp_r", "kd_r", "kdcomp_r")})
    C["gpos"] = {g: j for j, g in enumerate(C["genes"])}; C["tpos"] = {t: k for k, t in enumerate(C["targets"])}
    # own-genes M_total for the four-way set
    C["M_total_own"] = np.array([float(C["tot_r"][k, 0] @ C["tot_r"][k, 1]) if np.isfinite(C["tot_r"][k, 0, 0]) else np.nan
                                 for k in range(len(C["targets"]))])
    return C


def units(C, k, est):
    if est == "total":
        v = C["tot_r"][k]
    elif est == "within":
        v = C["tot_r"][k] - C["comp_r"][k]
    elif est == "kd":
        v = C["kd_r"][k]
    else:
        v = C["kd_r"][k] - C["kdcomp_r"][k]
    return None if not np.isfinite(v[0, 0]) else v


def compare(A, B, ess):
    shared_g = [g for g in A["genes"] if g in B["gpos"]]
    ia = np.array([A["gpos"][g] for g in shared_g]); ib = np.array([B["gpos"][g] for g in shared_g])
    rows = {}
    for t, ka in A["tpos"].items():
        kb = B["tpos"].get(t)
        if kb is None:
            continue
        r = {"essentiality": float(ess.get(t, np.nan)), "resid_a": float(A["resid"].get(t, np.nan)), "resid_b": float(B["resid"].get(t, np.nan))}
        for est in ESTS:
            va, vb = units(A, ka, est), units(B, kb, est)
            if va is None or vb is None:
                continue
            ua, ub = va[:, ia], vb[:, ib]
            Ma, Mb = float(ua[0] @ ua[1]), float(ub[0] @ ub[1])
            cross = float(ua.mean(0) @ ub.mean(0))
            r[est] = dict(rho=cross / np.sqrt(Ma * Mb) if Ma > 0 and Mb > 0 else np.nan, Ma=Ma, Mb=Mb)
        rows[t] = r
    return rows, len(shared_g)


def med_ci(x, rng):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 5:
        return dict(n=int(n), median=None, ci95=None)
    bs = np.array([np.median(x[rng.integers(0, n, n)]) for _ in range(NBOOT)])
    return dict(n=int(n), median=float(np.median(x)), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def summarize(rows, ts, a, b, rng):
    ts = [t for t in ts if a in rows[t] and b in rows[t] and np.isfinite(rows[t][a]["rho"]) and np.isfinite(rows[t][b]["rho"])]
    if len(ts) < 5:
        return dict(n=len(ts))
    ra = np.array([rows[t][a]["rho"] for t in ts]); rb = np.array([rows[t][b]["rho"] for t in ts])
    ma = np.array([rows[t][a]["Ma"] / rows[t][a]["Mb"] for t in ts]); mb = np.array([rows[t][b]["Ma"] / rows[t][b]["Mb"] for t in ts])
    out = dict(n=len(ts), rho_a=med_ci(ra, rng), rho_b=med_ci(rb, rng), delta=med_ci(rb - ra, rng), frac_improved=float(np.mean(rb > ra)))
    pos = (ma > 0) & (mb > 0)
    if pos.any():
        out["mag_ratio_a"] = float(2 ** np.median(np.log2(ma[pos]))); out["mag_ratio_b"] = float(2 ** np.median(np.log2(mb[pos])))
    return out


def main():
    dep, d, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True); ess = essentiality(dep)
    C = {}
    for spec in sys.argv[4:]:
        lab, ip = spec.split(":", 1); C[lab] = load(d, lab, ip)
        print("loaded", lab, "targets", len(C[lab]["targets"]), "genes", len(C[lab]["genes"]),
              "kd vectors", int(np.isfinite(C[lab]["kd_r"][:, 0, 0]).sum()), "residuals", len(C[lab]["resid"]), flush=True)
    labs = list(C); fourway = set.intersection(*[set(t for t, m in zip(C[l]["targets"], C[l]["M_total_own"]) if np.isfinite(m) and m > 0) for l in labs])
    print("four-way positive targets", len(fourway), flush=True)
    rng = np.random.default_rng(1)
    res = {"rules": __doc__, "labels": labs, "n_fourway": len(fourway), "pairs": {}}
    for a, b in itertools.combinations(labs, 2):
        rows, ng = compare(C[a], C[b], ess); name = "%s/%s" % (a, b)
        P = {"n_shared_genes": ng, "n_shared_targets": len(rows), "sets": {}}
        sets = {"all": list(rows), "fourway": [t for t in rows if t in fourway],
                "matched": [t for t in rows if rows[t]["resid_a"] <= KD and rows[t]["resid_b"] <= KD]}
        for sname, ts in sets.items():
            P["sets"][sname] = {}
            for nm, lo, hi in TIERS:
                tt = [t for t in ts if np.isfinite(rows[t]["essentiality"]) and lo <= rows[t]["essentiality"] < hi]
                P["sets"][sname][nm] = {"total_vs_within": summarize(rows, tt, "total", "within", rng),
                                        "total_vs_kd": summarize(rows, tt, "total", "kd", rng),
                                        "total_vs_kd_within": summarize(rows, tt, "total", "kd_within", rng)}
        res["pairs"][name] = P
        print("==", name, "genes", ng, "targets", len(rows), "matched", len(sets["matched"]), "fourway", len(sets["fourway"]), flush=True)
        for sname in sets:
            for nm in ("lt_-1", "pooled"):
                for comp in ("total_vs_within", "total_vs_kd"):
                    v = P["sets"][sname][nm][comp]
                    if v.get("n", 0) >= 5:
                        print("   %-8s %-7s %-16s n %4d rho %.3f -> %.3f  delta %.3f [%.3f, %.3f]  improved %.2f  mag %.2f -> %.2f" % (
                            sname, nm, comp, v["n"], v["rho_a"]["median"], v["rho_b"]["median"], v["delta"]["median"], v["delta"]["ci95"][0],
                            v["delta"]["ci95"][1], v["frac_improved"], v.get("mag_ratio_a", np.nan), v.get("mag_ratio_b", np.nan)), flush=True)
    # uniqueness index and ordering per estimator and set (pooled and most essential)
    res["ordering"] = {}
    for sname in ("all", "fourway", "matched"):
        for nm in ("lt_-1", "pooled"):
            for est_key, comp, side in (("total", "total_vs_within", "rho_a"), ("within", "total_vs_within", "rho_b"), ("kd", "total_vs_kd", "rho_b"),
                                        ("total_on_kd_set", "total_vs_kd", "rho_a")):
                vals = {}
                for name, P in res["pairs"].items():
                    v = P["sets"][sname][nm][comp]
                    if v.get("n", 0) >= 5:
                        vals[name] = v[side]["median"]
                if len(vals) < 6:
                    continue
                rp = [v for k, v in vals.items() if "RPE1" in k]; nr = [v for k, v in vals.items() if "RPE1" not in k]
                order = sorted(vals, key=lambda k: -vals[k])
                res["ordering"]["%s|%s|%s" % (sname, nm, est_key)] = dict(rho=vals, order=order, rpe1_uniqueness=float(np.mean(nr) - np.mean(rp)))
                print("order", sname, nm, est_key, " > ".join("%s %.3f" % (k, vals[k]) for k in order), "RPE1 uniqueness %.3f" % (np.mean(nr) - np.mean(rp)), flush=True)
    json.dump(res, open(out + "/metrics_nadig_pairs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
