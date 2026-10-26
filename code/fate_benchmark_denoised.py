"""C5 W5 (fate paper; expectations pre-stated in notes/fate_paper_plan.md §7, commit 3cde265): denoised benchmark,
run once per source context. Same code as the (bp) audit (code/a100/audit_bp_noise.py), which it copies.
Original audit header follows.
Method audit of findings (bp), fate paper C5 (2026-10-18). Read-only on fate_vectors_export.py outputs.
Question: fate_benchmark.py scores against the noisy ground truth (mean of two unit sides) and sets the scale factor
from ||truth|| / ||pred||. The knocked-down-cells truth uses a small fraction of each target's cells, so its noise
energy is larger. Noise inflates ||truth||, which pushes the scale factor up and pulls relative MSE toward 1. Are the
(bp) changes (scale 0.59 -> 0.97, relative MSE 3.18 -> 1.58) signal or noise?
Per target, for truth sides A, B and predictor sides PA, PB (source context, random halves):
  noise share of ||truth||^2 = ||(A - B) / 2||^2 / ||(A + B) / 2||^2
  denoised truth energy  S = A . B ;  denoised predictor energy  Q = PA . PB ;  cross  X = mean(P) . mean(T)
  denoised scale = sqrt(S / Q) ;  denoised relative MSE = (Q - 2X + S) / S ;  de-attenuated cosine = X / sqrt(Q S)
Same truth definitions, target subsets, gene alignment and unit choice as fate_benchmark.py. Medians per DepMap tier.
Usage: fate_benchmark_denoised.py <depmap csv> <vectors dir> <source> <target> [<target> ...]; writes
<vectors dir>/w5_benchmark_denoised_<source>.json (the K562-source run of the paper was made under the audit's name and
saved as audit_bp_noise_K562.json). Exploratory audit."""
import sys, json
import numpy as np
from fate_benchmark import essentiality, load, TIERS, GUIDE_CONTEXTS


def sides(C, k, kind):
    use_g = C["lab"] in GUIDE_CONTEXTS and np.isfinite(C["tot_g"][k, 0, 0])
    if kind in ("total", "within"):
        t, c = (C["tot_g"], C["comp_g"]) if use_g else (C["tot_r"], C["comp_r"])
        v = t[k] if kind == "total" else t[k] - c[k]
    else:
        # guide-unit contexts with guide-half knocked-down vectors: that ground truth on guide halves only, like the
        # total; no fallback to random halves (version 2, 2026-10-26)
        v = C["kd_g"][k] if C["lab"] in GUIDE_CONTEXTS and np.isfinite(C["kd_g"]).any() else C["kd_r"][k]
    return None if not np.isfinite(v[0, 0]) else v


def main():
    dep, d, src = sys.argv[1:4]; ess = essentiality(dep); S = load(d, src); res = {}
    tS = {t: k for k, t in enumerate(S["targets"])}; gS = {g: j for j, g in enumerate(S["genes"])}
    for tlab in sys.argv[4:]:
        X = load(d, tlab)
        shared = [g for g in X["genes"] if g in gS]
        ix = np.array([X["genes"].index(g) for g in shared]); isr = np.array([gS[g] for g in shared])
        common = {t for kx, t in enumerate(X["targets"]) if t in tS and sides(S, tS[t], "total") is not None
                  and all(sides(X, kx, k) is not None for k in ("total", "within", "kd"))}
        out = {}
        for kind in ("total", "kd"):
            for sub in ("all", "common"):
                per = {}
                for nm, lo, hi in TIERS:
                    rec = []
                    for kx, t in enumerate(X["targets"]):
                        e = ess.get(t, np.nan)
                        if t not in tS or not np.isfinite(e) or not (lo <= e < hi) or (sub == "common" and t not in common):
                            continue
                        T = sides(X, kx, kind); P = sides(S, tS[t], "total")
                        if T is None or P is None:
                            continue
                        A, B = T[0][ix], T[1][ix]; PA, PB = P[0][isr], P[1][isr]
                        tm, pm = (A + B) / 2, (PA + PB) / 2
                        Sx, Q, Xc = float(A @ B), float(PA @ PB), float(pm @ tm)
                        rec.append(dict(noise_share=float(((A - B) / 2) @ ((A - B) / 2) / max(tm @ tm, 1e-12)),
                                        naive_ratio=float(np.sqrt(tm @ tm) / max(np.sqrt(pm @ pm), 1e-12)),
                                        naive_rel_mse=float(((pm - tm) @ (pm - tm)) / max(tm @ tm, 1e-12)),
                                        S=Sx, Q=Q, X=Xc))
                    if len(rec) < 10:
                        per[nm] = dict(n=len(rec)); continue
                    Sv = np.array([r["S"] for r in rec]); Qv = np.array([r["Q"] for r in rec]); Xv = np.array([r["X"] for r in rec])
                    ok = (Sv > 0) & (Qv > 0)
                    per[nm] = dict(n=len(rec), n_pos=int(ok.sum()),
                                   noise_share_median=float(np.median([r["noise_share"] for r in rec])),
                                   naive_scale=float(np.median([r["naive_ratio"] for r in rec])),
                                   naive_rel_mse=float(np.median([r["naive_rel_mse"] for r in rec])),
                                   denoised_scale=float(np.median(np.sqrt(Sv[ok] / Qv[ok]))),
                                   denoised_rel_mse=float(np.median((Qv[ok] - 2 * Xv[ok] + Sv[ok]) / Sv[ok])),
                                   deatt_cosine=float(np.median(Xv[ok] / np.sqrt(Qv[ok] * Sv[ok]))))
                out["%s_%s" % (kind, sub)] = per
        res[tlab] = out
        for key, per in out.items():
            for nm, v in per.items():
                print(tlab, key, nm, json.dumps(v), flush=True)
    json.dump(res, open("%s/w5_benchmark_denoised_%s.json" % (d, src), "w"), indent=1)


if __name__ == "__main__":
    main()
