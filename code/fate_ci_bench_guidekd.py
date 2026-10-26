"""Intervals for Fig. 5b, c after the review fix of 2026-10-26: the "bs" block of fate_ci.py (denoised benchmark, hESC
most-essential stratum, common subset, per source; 2,000 target-resampled draws), with the hESC knocked-down ground
truth on guide halves (kd_g of fate_vectors_export.py) instead of random halves.
Usage: fate_ci_bench_guidekd.py <depmap csv> <vectors dir> <out json>. Exploratory."""
import sys, json
import numpy as np
from fate_crosscontext import essentiality, load, units
from fate_ci import tier_of, boot


def main():
    dep, d, outf = sys.argv[1:4]; ess = essentiality(dep); out = {"rules": __doc__}
    X = load(d, "hESC"); KG = np.load("%s/hESC_vectors.npz" % d)["kd_g"]
    for src in ("K562", "RPE1", "Jurkat", "HepG2"):
        Sv = load(d, src); gS = {g: j for j, g in enumerate(Sv["genes"])}; sg = [g for g in X["genes"] if g in gS]
        ix = np.array([X["genes"].index(g) for g in sg]); isr = np.array([gS[g] for g in sg]); tS = {t: j for j, t in enumerate(Sv["targets"])}
        rec = {"total": [], "kd": []}
        for kx, t in enumerate(X["targets"]):
            if tier_of(ess.get(t, np.nan)) != "lt_-1" or t not in tS:
                continue
            P_ = units(Sv, tS[t], "total"); Tt = units(X, kx, "total"); Tk = KG[kx]
            if P_ is None or Tt is None or not np.isfinite(Tk[0, 0]):
                continue
            PA, PB = P_[0][isr], P_[1][isr]; Q = float(PA @ PB); pm = (PA + PB) / 2
            for key, T in (("total", Tt), ("kd", Tk)):
                A, Bb = T[0][ix], T[1][ix]; Sx = float(A @ Bb); Xc = float(pm @ ((A + Bb) / 2))
                rec[key].append((Sx, Q, Xc))
        o = {}
        for key, v in rec.items():
            v = np.array(v); ok = (v[:, 0] > 0) & (v[:, 1] > 0); v = v[ok]
            o[key] = dict(denoised_scale=boot(lambda s, q: float(np.median(np.sqrt(s / q))), v[:, 0], v[:, 1]),
                          denoised_rel_mse=boot(lambda s, q, x: float(np.median((q - 2 * x + s) / s)), v[:, 0], v[:, 1], v[:, 2]))
        out[src] = o
        print(src, json.dumps(o), flush=True)
    json.dump(out, open(outf, "w"), indent=1)


if __name__ == "__main__":
    main()
