"""P4 item 1 (fate paper; expectations R1-R4 pre-stated in notes/fate_paper_plan.md, commit 0296375): the K562 budget
with the two INDEPENDENT screens as the two units (day-6 essential and day-8 genome-wide), instead of random cell
halves of one screen. Clones, library and harvest day all differ between the screens, so a cross-signal between them
carries no shared clone energy.
Per target, within each screen both random halves are pooled, giving one effect vector per screen, computed exactly as
fate_budget.vectors (phase-mixture composition counterfactual, +0.5 pseudocount over the library, response genes =
control >= 0.5 CP10K in BOTH screens, target gene zeroed):
  E_s = log2(P_t / P_c),  Ec_s = log2(P_comp / P_c),  Ew_s = E_s - Ec_s,  and the same from group 0 (knocked down)
Magnitudes across screens: M_total = E_d6 . E_d8, M_comp = Ec_d6 . Ec_d8, M_within = Ew_d6 . Ew_d8, M_kd likewise
(each screen's knocked-down effect has its own thinning null subtracted). Shares and the selection term as in
fate_budget.py; medians per DepMap tier, with the random-half numbers of each screen beside them.
Negative control: targets whose own gene is below 0.01 CP10K in both screens' controls.
Usage: fate_budget_screens.py <depmap csv> <extract dir A> <label A> <extract dir B> <label B> <out json>. Exploratory."""
import sys, json
import numpy as np
from fate_budget import Ctx, essentiality, control_cp10k, med, paired, TIERS, MIN_UNIT, ALL, MIN_RESP_CP10K


def pooled(C, k, groups, sel):
    """Effect and composition vectors for target index k, pooling both random halves of this screen."""
    A0, l0, n0 = C.sums(k, "random", 0, groups); A1, l1, n1 = C.sums(k, "random", 1, groups)
    Ac0, lc0, _ = C.sums(C.ct, "random", 0, ALL); Ac1, lc1, _ = C.sums(C.ct, "random", 1, ALL)
    A, lib, n = A0 + A1, l0 + l1, n0 + n1
    Ac, libc = Ac0 + Ac1, lc0 + lc1
    if n < MIN_UNIT or lib.sum() <= 0 or libc.sum() <= 0:
        return None
    qc = (Ac[:, sel] + 0.5) / np.maximum(libc, 1)[:, None]
    Pt = (A[:, sel].sum(0) + 0.5) / lib.sum()
    Pcomp = (lib / lib.sum()) @ qc
    Pc = (libc / libc.sum()) @ qc
    return np.log2(Pt / Pc), np.log2(Pcomp / Pc), n


def pooled_null(C, k, sel):
    """Thinning-null vector for target index k, both halves pooled."""
    if C.ncnt[k, :, :, 0].sum() < MIN_UNIT:
        return None
    A = np.asarray(C.nraw[k]).sum(axis=(0, 1)); lib = C.ncnt[k, :, :, 1].sum()
    Ac0, lc0, _ = C.sums(C.ct, "random", 0, ALL); Ac1, lc1, _ = C.sums(C.ct, "random", 1, ALL)
    Ac, libc = Ac0 + Ac1, lc0 + lc1
    qc = (Ac[:, sel] + 0.5) / np.maximum(libc, 1)[:, None]
    Pc = (libc / libc.sum()) @ qc
    return np.log2(((A[sel] + 0.5) / lib) / Pc)


def main():
    dep, dA, labA, dB, labB, outf = sys.argv[1:7]
    ess = essentiality(dep)
    A, B = Ctx(dA, labA), Ctx(dB, labB)
    cpA, cpB = control_cp10k(A), control_cp10k(B)
    gA = {g: q for q, g in enumerate(A.genes)}; gB = {g: q for q, g in enumerate(B.genes)}
    shared = [g for g in A.genes if g in gB and cpA[gA[g]] >= MIN_RESP_CP10K and cpB[gB[g]] >= MIN_RESP_CP10K]
    selA = np.array([gA[g] for g in shared]); selB = np.array([gB[g] for g in shared])
    posA = {q: j for j, q in enumerate(selA)}; posB = {q: j for j, q in enumerate(selB)}
    tA = {t: k for k, t in enumerate(A.tl)}; tB = {t: k for k, t in enumerate(B.tl)}
    both = [t for t in A.tl[:-1] if t in tB]
    print("shared response genes", len(shared), "shared targets", len(both), flush=True)
    rows = {}
    for t in both:
        ka, kb = tA[t], tB[t]
        va, vb = pooled(A, ka, ALL, selA), pooled(B, kb, ALL, selB)
        if va is None or vb is None:
            continue
        ja, jb = posA.get(gA.get(t, -1)), posB.get(gB.get(t, -1))
        Ea, Eca, na = va; Eb, Ecb, nb = vb
        for v, j in ((Ea, ja), (Eca, ja), (Eb, jb), (Ecb, jb)):
            if j is not None:
                v[j] = 0
        Ewa, Ewb = Ea - Eca, Eb - Ecb
        r = dict(m_total=float(Ea @ Eb), m_comp=float(Eca @ Ecb), m_within=float(Ewa @ Ewb),
                 essentiality=float(ess.get(t, np.nan)), n_d6=na, n_d8=nb,
                 resid_d6=float(A.idx["residual"].get(t, np.nan)), resid_d8=float(B.idx["residual"].get(t, np.nan)))
        ka_kd, kb_kd = pooled(A, ka, [0], selA), pooled(B, kb, [0], selB)
        nla, nlb = pooled_null(A, ka, selA), pooled_null(B, kb, selB)
        if all(x is not None for x in (ka_kd, kb_kd, nla, nlb)):
            Eka, Ekb = ka_kd[0] - nla, kb_kd[0] - nlb
            if ja is not None:
                Eka[ja] = 0
            if jb is not None:
                Ekb[jb] = 0
            r["m_kd"] = float(Eka @ Ekb)
        rows[t] = r
    out = {"rules": __doc__, "n_shared_genes": len(shared), "n_targets": len(rows), "tiers": {}}
    for nm, lo, hi in TIERS:
        ts = [t for t, r in rows.items() if np.isfinite(r["essentiality"]) and lo <= r["essentiality"] < hi]
        mt = np.array([rows[t]["m_total"] for t in ts]); mc = np.array([rows[t]["m_comp"] for t in ts])
        mw = np.array([rows[t]["m_within"] for t in ts]); mk = np.array([rows[t].get("m_kd", np.nan) for t in ts])
        pos = mt > 0
        out["tiers"][nm] = dict(n=len(ts), n_positive=int(pos.sum()), m_total_median=med(mt),
                                comp_share=med(mc[pos] / mt[pos]), within_share=med(mw[pos] / mt[pos]),
                                selection_kd_over_total=med(mk[pos] / mt[pos]), n_kd=int(np.isfinite(mk).sum()),
                                resid_d6=med([rows[t]["resid_d6"] for t in ts]), resid_d8=med([rows[t]["resid_d8"] for t in ts]),
                                cells_d6=med([rows[t]["n_d6"] for t in ts]), cells_d8=med([rows[t]["n_d8"] for t in ts]))
        print(nm, json.dumps(out["tiers"][nm]), flush=True)
    inert = [t for t in rows if t in gA and t in gB and cpA[gA[t]] < 0.01 and cpB[gB[t]] < 0.01]
    ne = out["tiers"]["gt_-0.2"]["m_total_median"]
    im = med([rows[t]["m_total"] for t in inert])
    out["negative_control"] = dict(n_inert=len(inert), inert_median_m_total=im, nonessential_median=ne,
                                   ratio=(im / ne) if (im is not None and ne) else None)
    print("negative control", json.dumps(out["negative_control"]), flush=True)
    out["rows"] = rows
    json.dump(out, open(outf, "w"), indent=1)


if __name__ == "__main__":
    main()
