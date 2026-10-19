"""Fate layer in iPSC (Feng et al. 2025 targeted CRISPRi, 19 HipSci lines, day 3; A100). Same metrics as pan_fate.py,
so iPSC sits next to hESC, CD4 T and the cancer lines; plus a per-line view: is the strength of elimination a donor
property (shared by the two clones of a donor)?
Pooled over lines, per target (>= MIN_CELLS): log n, dG1 vs NonTarget, response energy (two halves, expressed genes,
own target excluded), p53 target effect, knockdown residual (own raw UMI / (library x control rate)); medians in four
DepMap essentiality classes and Spearman with essentiality.
Per line c: depletion = median log(n_ct / N_c) of the most essential class minus that of the non-essential class (N_c =
assigned cells of line c); knockdown residual of the most essential class; energy ratio (most essential / non-essential).
Donor concordance: for the 9 donors with two clones, Spearman across donors between clone 1 and clone 2 of each per-line
metric, and the same-donor absolute difference vs different-donor absolute difference.
Usage: feng_fate.py <feng_agg> <out_dir>. Exploratory."""
import os, sys, json, itertools
import numpy as np
from scipy.stats import spearmanr
from pan_fate import P53, CLASSES
from strata import essentiality_of as essentiality

MIN_CELLS, MIN_EXPR = 20, 0.1


def per_target(S, c, n2, rows_of, gk, gkset, gpos, p53, ctrl, g1c, rate, T, ct, ess):
    tab = {k: [] for k in ("target", "n", "dG1", "energy", "p53_eff", "kd_resid", "ess")}
    for t, name in enumerate(T):
        if t == ct or n2[t].min() < MIN_CELLS // 2 or n2[t].sum() < MIN_CELLS:
            continue
        sA, sB = (sum(np.asarray(S[r]) for r in rows_of(t, h)) for h in range(2))
        hA = sA[gk] / n2[t, 0] - ctrl[0, gk]; hB = sB[gk] / n2[t, 1] - ctrl[1, gk]
        keep = np.ones(len(gk), bool)
        if name in gpos and gpos[name] in gkset:
            keep[np.where(gk == gpos[name])[0][0]] = False
        eff = (sA + sB)[gk] / n2[t].sum() - ctrl.mean(0)[gk]
        pk = np.isin(gk, p53) & keep
        kd = np.nan
        if name in gpos:
            lib = sum(c[r, 4] for h in range(2) for r in rows_of(t, h)); own = sum(c[r, 5] for h in range(2) for r in rows_of(t, h))
            exp = lib * rate[gpos[name]]
            if exp > 20:
                kd = own / exp
        g1 = sum(c[r, 1] for h in range(2) for r in rows_of(t, h)) / n2[t].sum()
        for k, v in (("target", name), ("n", n2[t].sum()), ("dG1", g1 - g1c), ("energy", float((hA * hB)[keep].sum())),
                     ("p53_eff", float(eff[pk].mean()) if pk.any() else np.nan), ("kd_resid", kd), ("ess", ess.get(name, np.nan))):
            tab[k].append(v)
    return {k: np.array(v) for k, v in tab.items()}


def classes(tab):
    e = tab["ess"]; ok = np.isfinite(e); res = {}
    for k in ("n", "dG1", "energy", "p53_eff", "kd_resid"):
        x = tab[k].astype(float); x = np.log(x) if k == "n" else x
        m = ok & np.isfinite(x); cl = []
        for lo, hi in CLASSES:
            mm = m & (e >= lo) & (e < hi)
            cl.append(dict(n=int(mm.sum()), median=float(np.median(x[mm])) if mm.sum() else None))
        res[k] = dict(rho_with_essentiality=float(spearmanr(-e[m], x[m])[0]), by_class=cl)
    return res


def main():
    agg, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    idx = json.load(open(agg + "/index.json")); T = idx["targets"]; L = idx["lines"]; genes = idx["genes"]
    S = np.load(agg + "/sums64.npy", mmap_mode="r"); c = np.load(agg + "/counts.npy")
    craw, clib = np.load(agg + "/ctrl_raw.npy"), np.load(agg + "/ctrl_lib.npy")
    nT, nL = len(T), len(L); ct = T.index("NonTarget"); gpos = {}
    for k, g in enumerate(genes):
        gpos.setdefault(g, k)
    row = lambda l, t, h: (l * nT + t) * 2 + h
    n = c[:, 0].reshape(nL, nT, 2)
    cm = np.load(agg + "/ctrl_mean.npy"); gk = np.where(cm >= MIN_EXPR)[0]; gkset = set(gk.tolist())
    p53 = np.array([gpos[g] for g in P53 if g in gpos and gpos[g] in gkset])
    ess = essentiality(set(T))
    # pooled over lines (each line's own control subtracted would need per-line effects; pooled control is used here
    # because every target is pooled over the same 19 lines in the same proportions as the control)
    n2 = n.sum(0)
    ctrl = np.stack([sum(np.asarray(S[row(l, ct, h)]) for l in range(nL)) / n2[ct, h] for h in range(2)])
    g1c = sum(c[row(l, ct, h), 1] for l in range(nL) for h in range(2)) / n2[ct].sum()
    rate = craw.sum(0) / max(clib.sum(), 1)
    tab = per_target(S, c, n2, lambda t, h: [row(l, t, h) for l in range(nL)], gk, gkset, gpos, p53, ctrl, g1c, rate, T, ct, ess)
    res = {"rules": __doc__, "pooled": dict(n_targets=int(len(tab["n"])), ctrl_cells=int(n2[ct].sum()), ctrl_G1=float(g1c),
                                           **classes(tab))}
    # per line
    per_line = {}
    for l in range(nL):
        nl = n[l]; Nl = nl.sum()
        ctl = np.stack([np.asarray(S[row(l, ct, h)]) / nl[ct, h] for h in range(2)])
        g1l = c[[row(l, ct, h) for h in range(2)], 1].sum() / nl[ct].sum()
        rl = craw[l] / max(clib[l], 1)
        tl = per_target(S, c, nl, lambda t, h, l=l: [row(l, t, h)], gk, gkset, gpos, p53, ctl, g1l, rl, T, ct, ess)
        e = tl["ess"]; lr = np.log(tl["n"] / Nl)
        top, bot = e < -1.0, e >= -0.2
        per_line[L[l]] = dict(n_targets=int(len(e)), ctrl_G1=float(g1l),
                              depletion=float(np.median(lr[top]) - np.median(lr[bot])),
                              kd_resid_essential=float(np.nanmedian(tl["kd_resid"][top])),
                              kd_resid_nonessential=float(np.nanmedian(tl["kd_resid"][bot])),
                              energy_ratio=float(np.median(tl["energy"][top]) / np.median(tl["energy"][bot])),
                              dG1_essential=float(np.median(tl["dG1"][top])), p53_essential=float(np.nanmedian(tl["p53_eff"][top])))
    res["per_line"] = per_line
    donors = {}
    for x in L:
        donors.setdefault(x.split("_")[0], []).append(x)
    two = {d: v for d, v in donors.items() if len(v) == 2}
    conc = {}
    for k in ("depletion", "kd_resid_essential", "energy_ratio", "ctrl_G1", "dG1_essential"):
        a = np.array([per_line[v[0]][k] for v in two.values()]); b = np.array([per_line[v[1]][k] for v in two.values()])
        same = np.abs(a - b)
        diff = [abs(per_line[x][k] - per_line[y][k]) for x, y in itertools.combinations(L, 2) if x.split("_")[0] != y.split("_")[0]]
        conc[k] = dict(rho_clone1_clone2=float(spearmanr(a, b)[0]), median_absdiff_same_donor=float(np.median(same)),
                       median_absdiff_diff_donor=float(np.median(diff)))
    lines_ = list(per_line)
    across = dict(rho_depletion_vs_kd_resid=float(spearmanr([per_line[x]["depletion"] for x in lines_],
                                                             [per_line[x]["kd_resid_essential"] for x in lines_])[0]),
                  rho_depletion_vs_energy_ratio=float(spearmanr([per_line[x]["depletion"] for x in lines_],
                                                                 [per_line[x]["energy_ratio"] for x in lines_])[0]))
    res["donor_concordance"] = conc; res["across_lines"] = across
    json.dump(res, open(out + "/metrics_feng_fate.json", "w"), indent=1, default=float)
    np.savez(out + "/pert_table_ipsc_pooled.npz", **tab)
    pr = lambda v: round(float(v), 3)
    print(json.dumps({k: v for k, v in res["pooled"].items()}, default=pr), flush=True)
    for x in lines_:
        print(x, {k: pr(v) for k, v in per_line[x].items()}, flush=True)
    print(json.dumps(conc, default=pr), json.dumps(across, default=pr), flush=True)


if __name__ == "__main__":
    main()
