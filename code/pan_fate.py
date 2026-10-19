"""Fate-layer test in hESC (Pan et al. 2026 genome-scale CRISPRi; A100), with the metrics of explore_W_fate.py so the
numbers sit next to K562 / RPE1 / Jurkat / HepG2 / CD4 T (Z1 d).
Question: Pan et al. report smaller perturbation effects in hESC than in K562 even for well knocked-down targets and
read this as hESC "resilience". The fate layer offers an alternative: strongly perturbed hESC are eliminated, so the
measured cells of essential targets are survivors with weak knockdown (as in primary CD4 T). Predictions:
  resilience  - essential targets: cell numbers ~ normal, knockdown residual low (good knockdown), response small;
  elimination - essential targets: cell numbers fall steeply with essentiality, knockdown residual rises, response
                diluted.
Per target (>= MIN_CELLS cells): n cells; dG1 vs non-targeting; response energy (two halves, expressed genes, own
target excluded); p53 target effect; knockdown residual = sum own raw UMI / (sum library x pooled control rate).
Essentiality: median DepMap 24Q4 gene effect over all lines; classes < -1, -1..-0.5, -0.5..-0.2, > -0.2.
Usage: pan_fate.py <pan_agg> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr
from strata import essentiality_of as essentiality

MIN_CELLS, MIN_EXPR = 20, 0.1
P53 = ["MDM2", "FDXR", "TP53I3", "RRM2B", "ZMAT3", "BAX", "CDKN1A", "GDF15", "SESN1", "SESN2", "TNFRSF10B", "AEN",
       "PLK3", "TP53INP1", "DDB2", "XPC", "RPS27L", "TRIAP1", "FAS", "BBC3", "PHLDA3", "SPATA18", "POLH", "TIGAR",
       "CCNG1", "APAF1", "EDA2R", "SULF2"]
CLASSES = ((-9, -1.0), (-1.0, -0.5), (-0.5, -0.2), (-0.2, 9))


def main():
    agg, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    idx = json.load(open(agg + "/index.json")); T = idx["targets"]; genes = idx["genes"]
    S = np.load(agg + "/sums64.npy", mmap_mode="r"); c = np.load(agg + "/counts.npy")
    craw, clib = np.load(agg + "/ctrl_raw.npy").sum(0), np.load(agg + "/ctrl_lib.npy").sum()
    ct = T.index(idx["ctrl"]); gpos = {g: k for k, g in enumerate(genes)}
    n2 = c[:, 0].reshape(len(T), 2)
    ctrl = np.stack([np.asarray(S[ct * 2 + h]) / n2[ct, h] for h in range(2)])
    gk = np.where(ctrl.mean(0) >= MIN_EXPR)[0]; gkset = set(gk.tolist())
    p53 = np.array([gpos[g] for g in P53 if g in gpos and gpos[g] in gkset])
    g1c = c[ct * 2:ct * 2 + 2, 1].sum() / n2[ct].sum()
    rate = craw / max(clib, 1)
    ess = essentiality(set(T))
    tab = {k: [] for k in ("target", "n", "dG1", "energy", "p53_eff", "kd_resid", "ess")}
    for t, name in enumerate(T):
        if t == ct or n2[t].min() < MIN_CELLS // 2 or n2[t].sum() < MIN_CELLS:
            continue
        hA = np.asarray(S[t * 2])[gk] / n2[t, 0] - ctrl[0, gk]; hB = np.asarray(S[t * 2 + 1])[gk] / n2[t, 1] - ctrl[1, gk]
        keep = np.ones(len(gk), bool)
        if name in gpos and gpos[name] in gkset:
            keep[np.where(gk == gpos[name])[0][0]] = False
        eff = (np.asarray(S[t * 2]) + np.asarray(S[t * 2 + 1]))[gk] / n2[t].sum() - ctrl.mean(0)[gk]
        pk = np.isin(gk, p53) & keep
        kd = np.nan
        if name in gpos:
            exp = c[t * 2:t * 2 + 2, 4].sum() * rate[gpos[name]]
            if exp > 20:
                kd = c[t * 2:t * 2 + 2, 5].sum() / exp
        for k, v in (("target", name), ("n", n2[t].sum()), ("dG1", c[t * 2:t * 2 + 2, 1].sum() / n2[t].sum() - g1c),
                     ("energy", float((hA * hB)[keep].sum())), ("p53_eff", float(eff[pk].mean()) if pk.any() else np.nan),
                     ("kd_resid", kd), ("ess", ess.get(name, np.nan))):
            tab[k].append(v)
    tab = {k: np.array(v) for k, v in tab.items()}
    e = tab["ess"]; ok = np.isfinite(e)
    res = {"rules": __doc__, "n_targets": int(len(e)), "n_with_depmap": int(ok.sum()), "ctrl_cells": int(n2[ct].sum()),
           "ctrl_G1": float(g1c)}
    for k in ("n", "dG1", "energy", "p53_eff", "kd_resid"):
        x = tab[k].astype(float); x = np.log(x) if k == "n" else x
        m = ok & np.isfinite(x)
        cls = []
        for lo, hi in CLASSES:
            mm = m & (e >= lo) & (e < hi)
            cls.append(dict(n=int(mm.sum()), median=float(np.median(x[mm])) if mm.sum() else None))
        res[k] = dict(rho_with_essentiality=float(spearmanr(-e[m], x[m])[0]), by_class=cls)
    np.savez(out + "/pert_table_hesc.npz", **tab)
    json.dump(res, open(out + "/metrics_pan_fate.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "rules"}, default=lambda v: round(float(v), 3)), flush=True)


if __name__ == "__main__":
    main()
