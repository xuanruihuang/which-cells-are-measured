"""Unified fate budget (fate paper P1; expectations T1-T5 pre-stated in notes/fate_paper_plan.md §3, commit 1f1901b).
Input: fate_budget_extract.py output for one or more contexts. Contexts are compared per essentiality tier, so their
gene sets need not match.
Per context, target t and unit side u (unit "random" = crc32 cell halves, available for every target; unit "guide" =
guide parity, only where the target has >= 2 guides), with the controls split by the same unit:
  v_t,k  library share of phase k among the target's cells;  q_c,k = (control raw + 0.5) / control library in phase k
  P_t    = (target raw + 0.5) / target library
  E = log2(P_t / P_c) = Ec + Ew,  Ec = log2((sum_k v_t,k q_c,k) / P_c)  (composition),  Ew = E - Ec  (within-cell)
  E_kd   the same from the knocked-down cells only (group 0); for the random design the thinning null of that unit side
         is subtracted (findings bi: the tail null is unusable, the thinning null is the matched one)
Magnitudes are cross-signals of the two unit sides: M_total = E_A . E_B, likewise M_comp, M_within, M_kd. Shares are
M_comp / M_total and M_within / M_total (cross terms make them not sum to 1); selection = M_kd / M_total.
T3: on the targets with >= 2 guides, both designs give the random-over-guide inflation per term and the share shift.
T4: everything recomputed with the Tirosh S / G2M genes removed from the vectors.
Response genes: control >= 0.5 CP10K only (fix of 10-18; see response_genes).
Negative control (every run): inert targets (target gene < 0.01 CP10K in controls) should give M_total ~0;
reported against the non-essential-tier median. Per-tier phase shifts dG1 / dS / dG2M (cell fractions).
Usage: fate_budget.py <depmap csv> <extract dir> <out_dir> <label> [<label> ...]. Exploratory."""
import os, sys, json
import numpy as np

TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf))
MIN_UNIT = 10
ALL = [0, 1, 2]
MIN_RESP_CP10K = 0.5


def essentiality(path):
    with open(path) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        rows = [np.array([float(x) if x else np.nan for x in line.rstrip("\n").split(",")[1:]]) for line in f]
    return dict(zip(sym, np.nanmedian(np.array(rows), 0)))


def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return float(2 ** np.median(np.log2(a[ok] / b[ok]))) if ok.any() else None


class Ctx:
    """raw[k] is (parity, half, phase, group, genes); cnt[k] the same keys x [n, lib, own, expected]."""

    def __init__(self, d, lab):
        self.idx = json.load(open("%s/%s_index.json" % (d, lab)))
        self.tl = self.idx["targets"]; self.genes = self.idx["genes"]
        T, G = len(self.tl), len(self.genes)
        self.raw = np.load("%s/%s_raw.npy" % (d, lab), mmap_mode="r").reshape(T, 2, 2, 3, 3, G)
        self.cnt = np.load("%s/%s_cnt.npy" % (d, lab)).reshape(T, 2, 2, 3, 3, 4)
        self.nraw = np.load("%s/%s_null_raw.npy" % (d, lab), mmap_mode="r")
        self.ncnt = np.load("%s/%s_null_cnt.npy" % (d, lab))
        self.gq = {g: q for q, g in enumerate(self.genes)}
        self.ct = T - 1
        self.gpt = self.idx["guides_per_target"]

    def sums(self, k, unit, u, groups):
        """(raw sums per phase x genes, library per phase, total cells) for target index k on unit side u."""
        A = np.asarray(self.raw[k]); C = self.cnt[k]
        if unit == "guide":
            A, C = A[u], C[u]                                  # (half, phase, group, genes)
        else:
            A, C = A[:, u], C[:, u]                            # (parity, phase, group, genes)
        A = A[:, :, groups, :].sum(axis=(0, 2))                # (phase, genes)
        lib = C[:, :, groups, 1].sum(axis=(0, 2))              # (phase,)
        n = C[:, :, groups, 0].sum()
        return A, lib, float(n)


def vectors(C, k, unit, u, groups, sel):
    """(E, Ec) over the selected gene positions for one unit side, or None if too few cells."""
    A, lib, n = C.sums(k, unit, u, groups)
    Ac, libc, nc = C.sums(C.ct, unit, u, ALL)
    if n < MIN_UNIT or libc.sum() <= 0 or lib.sum() <= 0:
        return None
    qc = (Ac[:, sel] + 0.5) / np.maximum(libc, 1)[:, None]
    Pt = (A[:, sel].sum(0) + 0.5) / lib.sum()
    Pcomp = (lib / lib.sum()) @ qc
    Pc = (libc / libc.sum()) @ qc
    return np.log2(Pt / Pc), np.log2(Pcomp / Pc)


def null_vector(C, k, u, sel):
    """Thinning-null effect vector (control cells classified knocked-down-like) for random-half side u."""
    if C.ncnt[k, :, u, 0].sum() < MIN_UNIT:
        return None
    A = np.asarray(C.nraw[k, :, u]).sum(0); lib = C.ncnt[k, :, u, 1].sum()
    Ac, libc, nc = C.sums(C.ct, "random", u, ALL)
    qc = (Ac[:, sel] + 0.5) / np.maximum(libc, 1)[:, None]
    Pc = (libc / libc.sum()) @ qc
    return np.log2(((A[sel] + 0.5) / lib) / Pc)


def control_cp10k(C):
    A, lib, n = C.sums(C.ct, "random", 0, ALL); B, libb, nb = C.sums(C.ct, "random", 1, ALL)
    return (A.sum(0) + B.sum(0)) / (lib.sum() + libb.sum()) * 1e4


def response_genes(C):
    """Genes used in the response vectors: control >= MIN_RESP_CP10K. The extraction also keeps every target gene (to
    count own-target UMIs); unexpressed ones must not enter the vectors: with near-zero counts the +0.5 pseudocount over
    a small target library vs the large control library gives log2 fold changes of ~+6, identical in both unit sides,
    which the cross-signal reads as a large reproducible response (bug found 10-18 on CD4 inert targets)."""
    A, lib, n = C.sums(C.ct, "random", 0, ALL); B, libb, nb = C.sums(C.ct, "random", 1, ALL)
    cp = (A.sum(0) + B.sum(0)) / (lib.sum() + libb.sum()) * 1e4
    return np.where(cp >= MIN_RESP_CP10K)[0]


def budget(C, ess, unit, drop_cycle=False):
    sel = response_genes(C)
    if drop_cycle:
        cyc = set(C.idx["cycle_genes"])
        sel = np.array([q for q in sel if C.genes[q] not in cyc])
    newpos = {q: j for j, q in enumerate(sel)}                 # old gene position -> position in sel
    rows = {}
    for k, t in enumerate(C.tl[:-1]):
        if unit == "guide" and C.gpt.get(t, 1) < 2:
            continue
        V = [vectors(C, k, unit, u, ALL, sel) for u in (0, 1)]
        if any(v is None for v in V):
            continue
        j = newpos.get(C.gq.get(t, -1))                        # target gene position after selection, or None
        E = [V[0][0], V[1][0]]; Ec = [V[0][1], V[1][1]]
        if j is not None:
            for v in E + Ec:
                v[j] = 0
        Ew = [E[0] - Ec[0], E[1] - Ec[1]]
        r = dict(m_total=float(E[0] @ E[1]), m_comp=float(Ec[0] @ Ec[1]), m_within=float(Ew[0] @ Ew[1]),
                 cells=int(C.cnt[k][..., 0].sum()), essentiality=float(ess.get(t, np.nan)),
                 resid=float(C.idx["residual"].get(t, np.nan)))
        K = [vectors(C, k, unit, u, [0], sel) for u in (0, 1)]
        if all(v is not None for v in K):
            Ek = [K[0][0], K[1][0]]
            if unit == "random":
                nl = [null_vector(C, k, u, sel) for u in (0, 1)]
                if all(v is not None for v in nl):
                    Ek = [Ek[0] - nl[0], Ek[1] - nl[1]]
            if j is not None:
                for v in Ek:
                    v[j] = 0
            r["m_kd"] = float(Ek[0] @ Ek[1])
        ph = C.cnt[k][..., 0].sum(axis=(0, 1, 3)); phc = C.cnt[C.ct][..., 0].sum(axis=(0, 1, 3))
        f, fc = ph / max(ph.sum(), 1), phc / max(phc.sum(), 1)
        r["dG1"], r["dS"], r["dG2M"] = (float(f[i] - fc[i]) for i in range(3))
        rows[t] = r
    return rows


def summarize(rows):
    out = {}
    for nm, lo, hi in TIERS:
        ts = [t for t, r in rows.items() if np.isfinite(r["essentiality"]) and lo <= r["essentiality"] < hi]
        mt = np.array([rows[t]["m_total"] for t in ts]); mc = np.array([rows[t]["m_comp"] for t in ts])
        mw = np.array([rows[t]["m_within"] for t in ts]); mk = np.array([rows[t].get("m_kd", np.nan) for t in ts])
        pos = mt > 0
        out[nm] = dict(n=len(ts), comp_share=med(mc[pos] / mt[pos]), within_share=med(mw[pos] / mt[pos]),
                       m_total_median=med(mt), selection_kd_over_total=med(mk[pos] / mt[pos]),
                       n_kd=int(np.isfinite(mk).sum()), resid=med([rows[t]["resid"] for t in ts]),
                       dG1=med([rows[t]["dG1"] for t in ts]), dS=med([rows[t]["dS"] for t in ts]),
                       dG2M=med([rows[t]["dG2M"] for t in ts]))
    return out


def main():
    dep, d, outd = sys.argv[1:4]; labs = sys.argv[4:]; os.makedirs(outd, exist_ok=True)
    ess = essentiality(dep)
    res = {"rules": __doc__}
    for lab in labs:
        C = Ctx(d, lab)
        rnd = budget(C, ess, "random"); gde = budget(C, ess, "guide"); nocyc = budget(C, ess, "random", drop_cycle=True)
        res[lab] = {"meta": dict(n_cells=C.idx["n_cells"], rkd=C.idx["rkd"], n_genes=len(C.genes), n_response_genes=int(len(response_genes(C))),
                                 n_targets_random=len(rnd), n_targets_guide=len(gde),
                                 n_cycle_genes=len(C.idx["cycle_genes"])),
                    "random": summarize(rnd), "guide": summarize(gde), "random_no_cycle_genes": summarize(nocyc)}
        both = sorted(set(rnd) & set(gde))
        cal = {}
        for nm, lo, hi in TIERS:
            ts = [t for t in both if np.isfinite(rnd[t]["essentiality"]) and lo <= rnd[t]["essentiality"] < hi]
            if not ts:
                continue
            cal[nm] = dict(n=len(ts),
                           inflation_total=paired([rnd[t]["m_total"] for t in ts], [gde[t]["m_total"] for t in ts]),
                           inflation_comp=paired([rnd[t]["m_comp"] for t in ts], [gde[t]["m_comp"] for t in ts]),
                           inflation_within=paired([rnd[t]["m_within"] for t in ts], [gde[t]["m_within"] for t in ts]),
                           comp_share_random=med([rnd[t]["m_comp"] / rnd[t]["m_total"] for t in ts if rnd[t]["m_total"] > 0]),
                           comp_share_guide=med([gde[t]["m_comp"] / gde[t]["m_total"] for t in ts if gde[t]["m_total"] > 0]))
        res[lab]["T3_calibration"] = cal
        # negative control: inert targets (target gene < 0.01 CP10K in the controls) should read ~0
        cpc = control_cp10k(C)
        inert = [t for t in rnd if t in C.gq and cpc[C.gq[t]] < 0.01]
        ne_med = res[lab]["random"]["gt_-0.2"]["m_total_median"]
        im = med([rnd[t]["m_total"] for t in inert])
        inert_g = [t for t in gde if t in C.gq and cpc[C.gq[t]] < 0.01]
        img = med([gde[t]["m_total"] for t in inert_g]); neg = res[lab]["guide"]["gt_-0.2"]["m_total_median"]
        res[lab]["negative_control"] = dict(n_inert=len(inert), inert_median_m_total=im,
                                            nonessential_median_m_total=ne_med,
                                            ratio=(im / ne_med) if (im is not None and ne_med) else None,
                                            guide_n_inert=len(inert_g), guide_inert_median_m_total=img,
                                            guide_nonessential_median_m_total=neg,
                                            guide_ratio=(img / neg) if (img is not None and neg) else None)
        print("  negative control", json.dumps(res[lab]["negative_control"]), flush=True)
        res[lab]["rows_random"] = rnd
        print("==", lab, json.dumps(res[lab]["meta"]), flush=True)
        for design in ("random", "guide", "random_no_cycle_genes"):
            for nm, v in res[lab][design].items():
                print("  %-22s %-10s %s" % (design, nm, json.dumps(v)), flush=True)
        for nm, v in cal.items():
            print("  calib %-10s %s" % (nm, json.dumps(v)), flush=True)
    json.dump(res, open(outd + "/metrics_fate_budget.json", "w"), indent=1)


if __name__ == "__main__":
    main()
