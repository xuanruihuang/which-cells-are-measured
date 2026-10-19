"""Fate redesign §6.2 (B): does the baseline escape fraction come from guide misassignment? (expectations B1-B2 and the
stratum control pre-stated, commit 0788b9a). A100. hESC only (guide UMIs per cell are available there).
Per analysable target (expected control count >= 3), its cells are split into tertiles of guide purity (top guide UMIs /
all guide UMIs), and separately of top-guide UMIs; the NB mixture of fate_mixture.py is fitted in each tertile with
>= 30 cells and in all cells. Reported per DepMap stratum: median e by tertile and for all cells.
Usage: fate_mixture_guideconf.py <percell_guideconf dir> <percell_hesc dir> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
from fate_mixture import compare, nb_size, MIN_MU, MIN_CELLS
from fate_budget import essentiality
import config

TIERS = (("lt_-1", -9, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, 9),
         ("gt_-1", -1.0, 9))


def main():
    dg, dh, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
    ess = essentiality(config.get("DEPMAP"))
    idx = json.load(open(dg + "/index.json")); T = idx["targets"]
    cand, lam = np.array(idx["cand"]), np.array(idx["lam_cand"])
    ch = np.load(dh + "/ctrl.npz"); assert list(ch["cand"]) == list(cand), "candidate order differs"
    O, lib = ch["O"], ch["lib"]; mean_lib = lib.mean()
    z = np.load(dg + "/cells.npz")
    tc, own, mu, top, tot = z["tc"], z["own"].astype(float), z["mu"], z["top"], z["tot"]
    pur = top / np.maximum(tot, 1)
    order = np.argsort(tc, kind="stable"); bounds = np.searchsorted(tc[order], np.arange(len(T) + 1))
    rows = []
    for j, t in enumerate(cand):
        if lam[j] * mean_lib < MIN_MU:
            continue
        r_ = order[bounds[t]:bounds[t + 1]]
        if len(r_) < 3 * MIN_CELLS:
            continue
        k = nb_size(O[:, j].astype(float), lam[j] * lib)
        row = dict(target=T[t], essentiality=float(ess.get(T[t], np.nan)), n=int(len(r_)))
        row["all"] = compare(own[r_], mu[r_], k)["e"]
        for name, score in (("purity", pur[r_]), ("top", top[r_])):
            rk = np.argsort(np.argsort(score + 1e-9 * np.random.default_rng(t).random(len(r_))))
            tert = (3 * rk) // len(r_)
            row[name] = [compare(own[r_][tert == q], mu[r_][tert == q], k)["e"] for q in range(3)]
            row[name + "_median_score"] = [float(np.median(score[tert == q])) for q in range(3)]
        rows.append(row)
    res = {"rules": __doc__, "n_targets": len(rows)}
    E = np.array([r["essentiality"] for r in rows])
    for lab, lo, hi in TIERS:
        m = np.isfinite(E) & (E >= lo) & (E < hi)
        if m.sum() < 5:
            res[lab] = dict(n=int(m.sum())); continue
        d = dict(n=int(m.sum()), e_all=float(np.median([rows[i]["all"] for i in np.where(m)[0]])))
        for name in ("purity", "top"):
            d[name] = [float(np.median([rows[i][name][q] for i in np.where(m)[0]])) for q in range(3)]
            d[name + "_high_over_all"] = d[name][2] / d["e_all"] if d["e_all"] > 0 else None
        res[lab] = d
        print(lab, json.dumps({a: (round(b, 3) if isinstance(b, float) else ([round(x, 3) for x in b] if isinstance(b, list) else b))
                               for a, b in d.items()}), flush=True)
    res["rows"] = rows
    json.dump(res, open(out + "/metrics_mixture_guideconf.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
