"""Composition vs within-cell decomposition of perturbation effects, RPE1 vs K562 (A100; fate paper C2; expectations
S1-S4 pre-stated in notes/fate_paper_plan.md §3, commit e1f14f4). Input: phase_decomp_extract.py output.
Per context, target t and random half h (controls split by the same halves):
  v_t,k  library share of phase k among the target's cells; q_c,k = (control raw + 0.5) / control library in phase k
  P_t    = (target raw + 0.5) / target library            (measured pseudobulk rate)
  P_comp = sum_k v_t,k q_c,k                              (phase mix of the target, control expression within phases)
  P_c    = sum_k v_c,k q_c,k
  E = log2(P_t / P_c) = Ec + Ew,  Ec = log2(P_comp / P_c) (composition),  Ew = log2(P_t / P_comp) (within-cell)
Genes: control >= 1 CP10K in both contexts; target gene zeroed. Magnitudes are half-0 x half-1 cross-signals:
M_total, M_comp, M_within (cross terms make the shares not sum to 1). dG1 = G1 cell fraction of the target minus the
controls'. Tiers: median DepMap 24Q4 gene effect over lines.
Usage: phase_decomp.py <extract dir> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr
from strata import essentiality, TIERS

MIN_HALF = 10


def load(path):
    z = np.load(path, allow_pickle=True); tl = [str(t) for t in z["targets"]]; K = len(tl)
    return dict(targets=tl, genes=[str(g) for g in z["genes"]], raw=z["raw"].astype(np.float64).reshape(K, 2, 3, -1),
                n=z["n"].reshape(K, 2, 3).astype(float), lib=z["lib"].reshape(K, 2, 3))


def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def decompose(D, keep, gq):
    c = len(D["targets"]) - 1; out = {}
    for k, t in enumerate(D["targets"][:-1]):
        E, Ec = [], []
        ok = True
        for h in (0, 1):
            if D["n"][k, h].sum() < MIN_HALF:
                ok = False; break
            vt = D["lib"][k, h] / D["lib"][k, h].sum(); vc = D["lib"][c, h] / D["lib"][c, h].sum()
            qc = (D["raw"][c, h][:, keep] + 0.5) / D["lib"][c, h][:, None]
            Pt = (D["raw"][k, h][:, keep].sum(0) + 0.5) / D["lib"][k, h].sum()
            Pcomp = vt @ qc; Pc = vc @ qc
            e = np.log2(Pt / Pc); ec = np.log2(Pcomp / Pc)
            if t in gq:
                e[gq[t]] = 0; ec[gq[t]] = 0
            E.append(e); Ec.append(ec)
        if not ok:
            continue
        Ew = [E[0] - Ec[0], E[1] - Ec[1]]
        g1 = D["n"][k, :, 0].sum() / D["n"][k].sum() - D["n"][c, :, 0].sum() / D["n"][c].sum()
        out[t] = dict(m_total=float(E[0] @ E[1]), m_comp=float(Ec[0] @ Ec[1]), m_within=float(Ew[0] @ Ew[1]), dG1=float(g1),
                      cells=int(D["n"][k].sum()))
    return out


def main():
    d, outd = sys.argv[1:3]; os.makedirs(outd, exist_ok=True)
    C = {lab: load("%s/%s.npz" % (d, lab)) for lab in ("K562", "RPE1")}
    assert C["K562"]["genes"] == C["RPE1"]["genes"] and C["K562"]["targets"] == C["RPE1"]["targets"]
    genes = C["K562"]["genes"]
    cp = [D["raw"][-1].sum((0, 1)) / D["lib"][-1].sum() * 1e4 for D in C.values()]
    keep = np.where((cp[0] >= 1) & (cp[1] >= 1))[0]; gq = {genes[j]: q for q, j in enumerate(keep)}
    ess = essentiality()
    R = {lab: decompose(D, keep, gq) for lab, D in C.items()}
    res = {"rules": __doc__, "n_genes": int(len(keep)), "n_targets": {k: len(v) for k, v in R.items()},
           "control_G1": {lab: float(D["n"][-1, :, 0].sum() / D["n"][-1].sum()) for lab, D in C.items()}}
    for lab, rows in R.items():
        T = {}
        for nm, lo, hi in TIERS:
            ts = [t for t in rows if np.isfinite(ess.get(t, np.nan)) and lo <= ess[t] < hi]
            mt = np.array([rows[t]["m_total"] for t in ts]); mc = np.array([rows[t]["m_comp"] for t in ts]); mw = np.array([rows[t]["m_within"] for t in ts])
            pos = mt > 0
            T[nm] = dict(n=len(ts), comp_share_median=med(mc[pos] / mt[pos]), within_share_median=med(mw[pos] / mt[pos]),
                         m_total_median=med(mt), m_comp_median=med(mc), m_within_median=med(mw),
                         comp_share_of_medians=(med(mc) / med(mt)) if med(mt) else None, dG1_median=med([rows[t]["dG1"] for t in ts]))
        res[lab] = T
        allt = [t for t in rows if rows[t]["m_total"] > 0]
        res[lab + "_S4_spearman_compshare_absdG1"] = float(spearmanr([rows[t]["m_comp"] / rows[t]["m_total"] for t in allt],
                                                                     [abs(rows[t]["dG1"]) for t in allt])[0])
    # S2: RPE1 / K562 paired ratio, total vs within, per tier
    s2 = {}
    for nm, lo, hi in TIERS:
        ts = [t for t in R["RPE1"] if t in R["K562"] and np.isfinite(ess.get(t, np.nan)) and lo <= ess[t] < hi]
        out = {"n": len(ts)}
        for kind in ("m_total", "m_within", "m_comp"):
            a = np.array([R["RPE1"][t][kind] for t in ts]); b = np.array([R["K562"][t][kind] for t in ts]); ok = (a > 0) & (b > 0)
            out[kind] = dict(n_both_positive=int(ok.sum()), ratio=float(2 ** np.median(np.log2(a[ok] / b[ok]))) if ok.any() else None)
        s2[nm] = out
    res["S2_rpe1_over_k562"] = s2
    res["rows"] = R
    json.dump(res, open(outd + "/metrics_phase_decomp.json", "w"), indent=1)
    for k, v in res.items():
        if k not in ("rules", "rows"):
            print(k, json.dumps(v), flush=True)


if __name__ == "__main__":
    main()
