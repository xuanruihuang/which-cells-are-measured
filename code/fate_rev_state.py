"""Fate v3r2 external review, check V11, analysis (expectation pre-stated in notes/fate_analysis_redesign.md §7.2, commit
f9d9a81). A100. Input: pan_extract_state.py output. The composition decomposition of fate_budget.py with pluripotency
state (three tertiles of the control score) in place of cell-cycle phase, guide parities as the two units:
  E = log2(P_t / P_c) = Ec + Ew, Ec = log2(sum_k v_t,k q_c,k / P_c), q_c,k the control rate in state k of that parity.
Response genes: control >= 0.5 CP10K; target gene zeroed. Per target: c = M_comp / M_total, w = M_within / M_total,
symmetric share (1 + c - w) / 2; medians per stratum with bootstrap intervals; and the shift in the share of the lowest-
pluripotency state relative to controls.
Usage: fate_rev_state.py <state dir> <out json>. Exploratory."""
import sys, json
import numpy as np
import config
from fate_budget import essentiality

DEPMAP = config.get("DEPMAP")
TIERS = (("lt_-1", -9, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, 9))
MIN_UNIT = 10


def bm(x, rng):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return None
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(2000)]
    return [float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(len(x))]


def main():
    d, outf = sys.argv[1:3]; rng = np.random.default_rng(17); ess = essentiality(DEPMAP)
    idx = json.load(open(d + "/index.json")); tl, genes = idx["targets"], idx["genes"]; T, G = len(tl), len(genes)
    R = np.load(d + "/raw_sums.npy", mmap_mode="r").reshape(T, 2, 3, G); C = np.load(d + "/counts.npy").reshape(T, 2, 3, 2)
    ct = tl.index("non-targeting")
    cp = np.asarray(R[ct]).sum(axis=(0, 1)) / C[ct, :, :, 1].sum() * 1e4
    sel = np.where(cp >= 0.5)[0]; gq = {g: j for j, g in enumerate(genes[i] for i in sel)}
    Rc = np.asarray(R[ct])[:, :, sel]; Cc = C[ct]
    fc = Cc[:, :, 1].sum(0) / Cc[:, :, 1].sum()                    # control library share per state
    rows = []
    for k, t in enumerate(tl):
        if k == ct or C[k, 0, :, 0].sum() < MIN_UNIT or C[k, 1, :, 0].sum() < MIN_UNIT:
            continue
        E, Ec = [], []
        for p in (0, 1):
            A = np.asarray(R[k, p])[:, sel]; lib = C[k, p, :, 1]
            qc = (Rc[p] + 0.5) / np.maximum(Cc[p, :, 1], 1)[:, None]
            Pt = (A.sum(0) + 0.5) / lib.sum(); Pc = (Cc[p, :, 1] / Cc[p, :, 1].sum()) @ qc; Pcomp = (lib / lib.sum()) @ qc
            e, ec = np.log2(Pt / Pc), np.log2(Pcomp / Pc)
            j = gq.get(t)
            if j is not None:
                e[j] = 0; ec[j] = 0
            E.append(e); Ec.append(ec)
        Ew = [E[0] - Ec[0], E[1] - Ec[1]]
        f = C[k, :, :, 1].sum(0) / C[k, :, :, 1].sum()
        rows.append(dict(target=t, essentiality=float(ess.get(t, np.nan)), m_total=float(E[0] @ E[1]), m_comp=float(Ec[0] @ Ec[1]),
                         m_within=float(Ew[0] @ Ew[1]), d_low_state=float(f[0] - fc[0]), d_high_state=float(f[2] - fc[2]),
                         cells=int(C[k, :, :, 0].sum())))
    res = {"rules": __doc__, "n_genes": int(len(sel)), "control_state_library_share": [float(x) for x in fc], "tiers": {}}
    for nm, lo, hi in TIERS:
        rr = [r for r in rows if np.isfinite(r["essentiality"]) and lo <= r["essentiality"] < hi]
        pos = [r for r in rr if r["m_total"] > 0]
        c = np.array([r["m_comp"] / r["m_total"] for r in pos]); w = np.array([r["m_within"] / r["m_total"] for r in pos])
        res["tiers"][nm] = dict(n=len(rr), n_pos=len(pos), comp=bm(c, rng), within=bm(w, rng), symmetric=bm((1 + c - w) / 2, rng),
                                d_low_state=bm([r["d_low_state"] for r in rr], rng), d_high_state=bm([r["d_high_state"] for r in rr], rng))
        print(nm, json.dumps(res["tiers"][nm]), flush=True)
    res["rows"] = rows
    json.dump(res, open(outf, "w"), indent=1)


if __name__ == "__main__":
    main()
