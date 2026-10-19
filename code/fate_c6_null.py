"""C6 failing control V4 (fate paper; expectations pre-stated in notes/fate_paper_plan.md §7, commit 3cde265).
Subtracting any reproducible, reference-orthogonal vector raises the disattenuated correlation rho, so the Jurkat
control of findings (bo) could not fail. Here the composition vector subtracted from a target is replaced by another
target's: (a) own (E - Ec, as bo), (b) a random other target of the same DepMap tier, (c) a random target of the
non-essential tier (DepMap > -0.2). Each unit side subtracts the matching unit side of the donor target. The reference
(K562) is always its own within-cell estimate, as in bo. Same shared targets / genes / rho as fate_crosscontext.py.
Per target: delta rho = rho(subtracted) - rho(total); (b) and (c) use the median over PERM donors.
Usage: fate_c6_null.py <depmap csv> <vectors dir> <reference> <label> [<label> ...]. Exploratory."""
import sys, json
import numpy as np
from fate_crosscontext import essentiality, load, units, TIERS

PERM = 200


def rho(ux, ur):
    Mx, Mr = float(ux[0] @ ux[1]), float(ur[0] @ ur[1])
    return float(ux.mean(0) @ ur.mean(0)) / np.sqrt(Mx * Mr) if Mx > 0 and Mr > 0 else np.nan


def main():
    dep, d, ref = sys.argv[1:4]; ess = essentiality(dep); R = load(d, ref); res = {"rules": __doc__}
    rng = np.random.default_rng(3)
    for lab in sys.argv[4:]:
        X = load(d, lab)
        gR = {g: j for j, g in enumerate(R["genes"])}; shared = [g for g in X["genes"] if g in gR]
        ix = np.array([X["genes"].index(g) for g in shared]); ir = np.array([gR[g] for g in shared])
        tR = {t: k for k, t in enumerate(R["targets"])}
        tot, comp, refw, tier = {}, {}, {}, {}
        for kx, t in enumerate(X["targets"]):
            if t not in tR:
                continue
            vt, vw = units(X, kx, "total"), units(X, kx, "within")
            rw = units(R, tR[t], "within")
            if vt is None or vw is None or rw is None:
                continue
            tot[t] = vt[:, ix]; comp[t] = (vt - vw)[:, ix]; refw[t] = rw[:, ir]
            e = ess.get(t, np.nan)
            tier[t] = next((nm for nm, lo, hi in TIERS if np.isfinite(e) and lo <= e < hi), None)
        noness = [t for t in tot if tier[t] == "gt_-0.2"]
        out = {}
        for nm in ("lt_-1", "-1_-0.5"):
            ts = [t for t in tot if tier[t] == nm]
            d_own, d_same, d_cross, ok = [], [], [], []
            for t in ts:
                r0 = rho(tot[t], refw[t]); ra = rho(tot[t] - comp[t], refw[t])
                others = [u for u in ts if u != t]
                rb = [rho(tot[t] - comp[others[i]], refw[t]) for i in rng.integers(0, len(others), PERM)]
                rc = [rho(tot[t] - comp[noness[i]], refw[t]) for i in rng.integers(0, len(noness), PERM)]
                if not (np.isfinite(r0) and np.isfinite(ra)):
                    continue
                d_own.append(ra - r0); d_same.append(np.nanmedian(rb) - r0); d_cross.append(np.nanmedian(rc) - r0)
            d_own, d_same, d_cross = map(np.array, (d_own, d_same, d_cross))
            out[nm] = dict(n=len(d_own), delta_own=float(np.median(d_own)), delta_same_tier_mismatch=float(np.nanmedian(d_same)),
                           delta_nonessential_mismatch=float(np.nanmedian(d_cross)),
                           own_minus_same=float(np.nanmedian(d_own - d_same)),
                           frac_own_gt_same=float(np.nanmean(d_own > d_same)),
                           comp_norm_ratio_own_over_nonessential=float(np.median([np.sqrt(abs(comp[t][0] @ comp[t][1])) for t in ts]) /
                                                                       max(np.median([np.sqrt(abs(comp[t][0] @ comp[t][1])) for t in noness]), 1e-12)))
            print(lab, nm, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[nm].items()}), flush=True)
        res[lab] = out
    json.dump(res, open("%s/metrics_c6_null_%s.json" % (d, ref), "w"), indent=1)


if __name__ == "__main__":
    main()
