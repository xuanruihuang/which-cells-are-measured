"""Fate paper, redesign section 3.1: how much of a genome-wide library carries a heavy selection distortion. (5090.)
Contexts: hESC (Pan 2026), K562 genome-wide day 8 (Replogle 2022, part 01), CD4 T donor 1 resting (Zhu 2026).
Each target recovered in the data falls in one class:
  (i)  fewer than 20 cells: depleted to the point of not being measurable;
  (ii) residual not estimable (target gene absent or expected count <= 20; CD4 table: residual not finite);
  (iii) predicted dilution d = (1 - e)^2 < 0.5, with escape fraction e = (R - r_kd) / (1 - r_kd) clipped to [0, 0.95],
       R the survivor residual and r_kd the median residual of the non-essential stratum (DepMap > -0.2);
  (iv) d >= 0.5.
Heavy distortion = (i) + (iii). Essentiality: DepMap 24Q4 median gene effect over all lines, four strata.
Usage: fate_libshare_summary.py <dir>  (dir holds k562gw_residual.npz, pert_table_hesc.npz, pan_index.json,
pan_counts.npy; the CD4 tables are the CD4_PERT_TABLE and CD4_PSEUDOBULK settings)."""
import sys, json
import numpy as np
import config

DEPMAP = config.get("DEPMAP")
CD4 = config.get("CD4_PERT_TABLE")
CD4_PB = config.get("CD4_PSEUDOBULK")
STRATA = (("< -1", -9, -1.0), ("-1 to -0.5", -1.0, -0.5), ("-0.5 to -0.2", -0.5, -0.2), ("> -0.2", -0.2, 9))
MIN_CELLS = 20


def depmap_median():
    with open(DEPMAP) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        rows = [np.array([float(x) if x else np.nan for x in line.rstrip("\n").split(",")[1:]]) for line in f]
    return dict(zip(sym, np.nanmedian(np.array(rows), 0)))


def classify(name, n, R, ess):
    e_ = np.array([ess.get(t, np.nan) for t in name])
    non = np.isfinite(e_) & (e_ > -0.2) & (n >= MIN_CELLS) & np.isfinite(R)
    r_kd = float(np.median(R[non]))
    e = np.clip((R - r_kd) / (1 - r_kd), 0, 0.95)
    d = (1 - e) ** 2
    cls = np.where(n < MIN_CELLS, 1, np.where(~np.isfinite(R), 2, np.where(d < 0.5, 3, 4)))
    out = {"r_kd": r_kd, "n_targets": int(len(name)), "n_with_depmap": int(np.isfinite(e_).sum())}

    def frac(m):
        k = int(m.sum())
        return dict(n=k, **{c: float((cls[m] == j).mean()) if k else None
                            for j, c in ((1, "i_unmeasurable"), (2, "ii_not_estimable"), (3, "iii_d_below_half"),
                                         (4, "iv_d_at_least_half"))},
                    heavy=float(np.isin(cls[m], (1, 3)).mean()) if k else None)
    out["all"] = frac(np.ones(len(name), bool))
    out["strata"] = {}
    for lab, lo, hi in STRATA:
        m = np.isfinite(e_) & (e_ >= lo) & (e_ < hi)
        out["strata"][lab] = dict(frac(m), share_of_library=float(m.mean()))
    out["no_depmap"] = frac(~np.isfinite(e_))
    return out


def main():
    d = sys.argv[1]
    ess = depmap_median()
    res = {"rules": __doc__}
    k = np.load(d + "/k562gw_residual.npz", allow_pickle=True)
    res["K562_gw_day8"] = classify(k["target"], k["n"], k["kd_resid"], ess)
    idx = json.load(open(d + "/pan_index.json")); T = idx["targets"]; c = np.load(d + "/pan_counts.npy")
    ct = T.index(idx["ctrl"]); n_all = c[:, 0].reshape(len(T), 2).sum(1)
    h = np.load(d + "/pert_table_hesc.npz", allow_pickle=True)
    Rh = dict(zip(h["target"], h["kd_resid"]))
    names = np.array([t for i, t in enumerate(T) if i != ct and n_all[i] > 0])
    nn = np.array([n_all[i] for i, t in enumerate(T) if i != ct and n_all[i] > 0], float)
    R = np.array([Rh.get(t, np.nan) for t in names], float)
    res["hESC"] = classify(names, nn, R, ess)
    z = np.load(CD4, allow_pickle=True)
    nm = z["group"] if "group" in z.files else z["pert"]
    tab = dict(zip(nm, zip(z["n"], z["kd_resid"])))
    pb = np.load(CD4_PB, allow_pickle=True)["perts"]
    allp = sorted(set(nm) | set(pb) - {"non-targeting", "NTC", "CTRL"})
    nn = np.array([tab[p][0] if p in tab else 0 for p in allp], float)
    R = np.array([tab[p][1] if p in tab else np.nan for p in allp], float)
    res["CD4_D1_rest"] = classify(np.array(allp), nn, R, ess)
    json.dump(res, open(d + "/metrics_libshare.json", "w"), indent=1)
    for ctx in ("hESC", "K562_gw_day8", "CD4_D1_rest"):
        r = res[ctx]
        print(ctx, "r_kd %.3f" % r["r_kd"], "n", r["n_targets"], "all", json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in r["all"].items()}))
        for lab, v in r["strata"].items():
            print("   ", lab, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()}))


if __name__ == "__main__":
    main()
