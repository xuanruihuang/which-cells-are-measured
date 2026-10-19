"""Fate over time in Drepanos et al. 2026 K562 CRISPRi vs CRISPRko (5090; fate paper C3; expectations D1-D3 in
notes/fate_paper_plan.md §3, committed before running). About 25 targets, 4 guides per modality, 10 intergenic control
guides shared by both modalities ("Both"); cells sampled on days 4, 7, 10, 14 (day 14 stored as separate CRISPRi and
CRISPRko objects; other days hold both modalities in one object). Raw counts (cells x genes, CSC).
Per day, modality and target:
  abundance: cells of the target / cells of the non-essential targets of the same modality (DepMap K562 > -0.2), divided
    by the same ratio on day 4 (primary; does not depend on how the shared controls split between modalities);
    secondary: cells / intergenic control cells, relative to day 4;
  knockdown residual: own-target raw UMIs / (control rate x library), control rate from intergenic cells of that day;
  response: guide halves (the target's guides sorted by name, alternate assignment) as independent units; effect =
    log2 CPM fold change of the half's pseudobulk sum against half of the control guides (5 vs 5, disjoint), genes with
    control >= 1 CP10K, target zeroed; response = cross-signal of the two halves. Null: control half A vs half B
    split again into guide pairs (control guides against the other control guides).
Essentiality: DepMap 24Q4 K562 (ACH-000551); essential < -0.5, non-essential > -0.2.
Revision after the first run (2026-10-18): on days 4, 7, 10 the intergenic controls are shared by the CRISPRi and
CRISPRko cell populations, which differ in baseline expression (an expression classifier separates the two arms at 79%
vs 58% base rate; cell barcodes do not). Effects against these controls therefore carry the arm difference, a component
shared by every target (non-essential targets read 5-25 on days 4-10 and ~0 on day 14, where the arms are stored
separately). Primary response is now arm-matched: reference = targets of the same modality with DepMap > -0.2, day-14
abundance 0.7-1.4 and day-14 |response vs controls| < 2 (selection uses only day-14 control-based numbers); each
target's halves are compared with the reference cells of matching guide parity, the target itself left out; residual
likewise uses the reference cells' rate. Null: each reference target against the remaining references.
Usage: fate_drepanos.py <drepanos dir> <depmap csv> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
import h5py
from scipy.sparse import csc_matrix, csr_matrix

DAYS = (4, 7, 10, 14)
MODS = ("CRISPRi", "CRISPRko")
MIN_CP10K = 1.0
MIN_CELLS = 20


def k562_effect(path):
    with open(path) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        for line in f:
            if line.startswith("ACH-000551,"):
                v = [float(x) if x else np.nan for x in line.rstrip("\n").split(",")[1:]]
                return dict(zip(sym, v))


def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def load(path, grp):
    with h5py.File(path) as f:
        c = f[grp]
        shape = tuple(c["shape"][:])
        M = csc_matrix((c["data"][:].astype(np.float32), c["indices"][:], c["indptr"][:]), shape=shape).tocsr()
        genes = dec(c["genes"][:]); guide = dec(c["obs"]["Guide"][:]); mod = dec(c["obs"]["modality"][:])
    return M, genes, guide, mod


def lfc(x, c):
    return np.log2((x / x.sum() * 1e6 + 1) / (c / c.sum() * 1e6 + 1))


def group_sum(M, idx):
    return np.asarray(M[idx].sum(0)).ravel().astype(np.float64)


def day_metrics(M, genes, guide, mod, use_mod):
    lib = np.asarray(M.sum(1)).ravel()
    gpos = {g: j for j, g in enumerate(genes)}
    isc = np.char.startswith(guide.astype(str), "INTERGENIC")
    tgt = np.where(isc, "CTRL", np.array([g.split("_")[0] for g in guide]))
    cg = sorted(set(guide[isc])); cA = np.isin(guide, cg[0::2]); cB = np.isin(guide, cg[1::2])
    ctrl_sum = group_sum(M, np.where(isc)[0]); lam = ctrl_sum / lib[isc].sum()
    keep = np.where(ctrl_sum / ctrl_sum.sum() * 1e4 >= MIN_CP10K)[0]; kset = {j: q for q, j in enumerate(keep)}
    SA, SB = group_sum(M, np.where(cA)[0])[keep], group_sum(M, np.where(cB)[0])[keep]
    # null: control guides 0,2,4 vs control half A(other) ... use guide pairs within controls
    null = []
    for k in range(0, len(cg) - 1, 2):
        a = np.where(guide == cg[k])[0]; b = np.where(guide == cg[k + 1])[0]
        restA = np.where(np.isin(guide, [g for i, g in enumerate(cg) if i % 2 == 0 and i != k]))[0]
        restB = np.where(np.isin(guide, [g for i, g in enumerate(cg) if i % 2 == 1 and i != k + 1]))[0]
        null.append(float(lfc(group_sum(M, a)[keep], group_sum(M, restA)[keep]) @ lfc(group_sum(M, b)[keep], group_sum(M, restB)[keep])))
    out = {}
    for t in sorted(set(tgt[(mod == use_mod)]) - {"CTRL"}):
        m = (tgt == t) & (mod == use_mod)
        gl = sorted(set(guide[m]))
        rec = dict(cells=int(m.sum()), guides=len(gl), ctrl_cells=int(isc.sum()))
        j = gpos.get(t)
        rec["residual"] = float(group_sum(M, np.where(m)[0])[j] / (lam[j] * lib[m].sum())) if j is not None and lam[j] > 0 else None
        rec["control_cp10k"] = float(lam[j] * 1e4) if j is not None else None
        if len(gl) >= 2:
            hA = np.where(m & np.isin(guide, gl[0::2]))[0]; hB = np.where(m & np.isin(guide, gl[1::2]))[0]
            if len(hA) >= MIN_CELLS and len(hB) >= MIN_CELLS:
                a = lfc(group_sum(M, hA)[keep], SA); b = lfc(group_sum(M, hB)[keep], SB)
                if j in kset:
                    a[kset[j]] = 0; b[kset[j]] = 0
                rec["response"] = float(a @ b); rec["cells_halves"] = [int(len(hA)), int(len(hB))]
        out[t] = rec
    return out, dict(n_genes=int(len(keep)), null=null, umi_per_cell=float(np.median(lib)))


def ref_metrics(M, genes, guide, mod, use_mod, refs):
    """Arm-matched effects: reference = cells of the reference targets (same modality), split by guide parity."""
    lib = np.asarray(M.sum(1)).ravel(); gpos = {g: j for j, g in enumerate(genes)}
    sel = mod == use_mod
    tgt = np.array([g.split("_")[0] for g in guide])
    par = np.full(len(guide), -1)
    for t in set(tgt[sel]):
        gl = sorted(set(guide[sel & (tgt == t)]))
        for r, g in enumerate(gl):
            par[sel & (guide == g)] = r % 2
    isref = sel & np.isin(tgt, refs)
    out, null = {}, {}
    sums = {}
    for t in sorted(set(tgt[sel])):
        for p in (0, 1):
            ix = np.where(sel & (tgt == t) & (par == p))[0]
            sums[(t, p)] = (group_sum(M, ix), len(ix), lib[ix].sum())
    G = len(genes)
    for t in sorted(set(tgt[sel])):
        others = [r for r in refs if r != t and (r, 0) in sums]
        RA = sum(sums[(r, 0)][0] for r in others); RB = sum(sums[(r, 1)][0] for r in others)
        Rall = RA + RB; Rlib = sum(sums[(r, p)][2] for r in others for p in (0, 1))
        keep = np.where(Rall / Rall.sum() * 1e4 >= MIN_CP10K)[0]; kset = {j: q for q, j in enumerate(keep)}
        j = gpos.get(t)
        rec = dict(cells=int(sum(sums[(t, p)][1] for p in (0, 1))))
        own = sums[(t, 0)][0] + sums[(t, 1)][0]; tl = sums[(t, 0)][2] + sums[(t, 1)][2]
        rec["residual_ref"] = float(own[j] / (Rall[j] / Rlib * tl)) if j is not None and Rall[j] > 0 else None
        if min(sums[(t, 0)][1], sums[(t, 1)][1]) >= MIN_CELLS:
            a = lfc(sums[(t, 0)][0][keep], RA[keep]); b = lfc(sums[(t, 1)][0][keep], RB[keep])
            if j in kset:
                a[kset[j]] = 0; b[kset[j]] = 0
            rec["response_ref"] = float(a @ b)
        out[t] = rec
        if t in refs and "response_ref" in rec:
            null[t] = rec["response_ref"]
    return out, null


def main():
    d, dep, outd = sys.argv[1:4]; os.makedirs(outd, exist_ok=True)
    ess = k562_effect(dep)
    res = {"rules": __doc__, "days": {}}
    for day in DAYS:
        path = "%s/K562_Day%d.h5" % (d, day)
        with h5py.File(path) as f:
            grps = [k for k in f if k.startswith("counts")]
        res["days"][day] = {}
        for mo in MODS:
            grp = [g for g in grps if g.endswith(mo)] or ["counts"]
            M, genes, guide, mod = load(path, grp[0])
            keepc = (mod == mo) | (mod == "Both")
            M, guide, mod = M[np.where(keepc)[0]], guide[keepc], mod[keepc]
            rec, meta = day_metrics(M, genes, guide, mod, mo)
            res["days"][day][mo] = dict(meta=meta, targets=rec)
            print(day, mo, json.dumps(meta), flush=True)
            del M
    # reference targets from day-14 control-based numbers (arms stored separately on day 14)
    refs = {}
    for mo in MODS:
        T14, T4 = res["days"][14][mo]["targets"], res["days"][4][mo]["targets"]
        ne = [t for t in T14 if t in T4 and ess.get(t, np.nan) > -0.2]
        ref0 = sum(T4[t]["cells"] for t in ne); ref14 = sum(T14[t]["cells"] for t in ne)
        refs[mo] = sorted(t for t in ne if "response" in T14[t] and abs(T14[t]["response"]) < 2
                          and 0.7 <= (T14[t]["cells"] / ref14) / (T4[t]["cells"] / ref0) <= 1.4)
        print("reference", mo, refs[mo], flush=True)
    res["reference_targets"] = refs
    for day in DAYS:
        path = "%s/K562_Day%d.h5" % (d, day)
        with h5py.File(path) as f:
            grps = [k for k in f if k.startswith("counts")]
        for mo in MODS:
            grp = [g for g in grps if g.endswith(mo)] or ["counts"]
            M, genes, guide, mod = load(path, grp[0])
            keepc = mod == mo
            M, guide, mod = M[np.where(keepc)[0]], guide[keepc], mod[keepc]
            rr, nl = ref_metrics(M, genes, guide, mod, mo, refs[mo])
            for t, v in rr.items():
                res["days"][day][mo]["targets"].setdefault(t, {}).update(v)
            res["days"][day][mo]["meta"]["null_ref"] = nl
            print(day, mo, "null_ref", json.dumps(nl), flush=True)
            del M
    # abundance relative to day 4
    for mo in MODS:
        tg = sorted(set.intersection(*[set(res["days"][dd][mo]["targets"]) for dd in DAYS]))
        ne = [t for t in tg if ess.get(t, np.nan) > -0.2]
        for dd in DAYS:
            T = res["days"][dd][mo]["targets"]; ref = sum(T[t]["cells"] for t in ne)
            T0 = res["days"][4][mo]["targets"]; ref0 = sum(T0[t]["cells"] for t in ne)
            c0 = res["days"][4][mo]["targets"][tg[0]]["ctrl_cells"]
            for t in tg:
                T[t]["abundance_vs_nonessential"] = float((T[t]["cells"] / ref) / (T0[t]["cells"] / ref0))
                T[t]["abundance_vs_control"] = float((T[t]["cells"] / T[t]["ctrl_cells"]) / (T0[t]["cells"] / T0[t]["ctrl_cells"]))
                T[t]["essentiality"] = float(ess.get(t, np.nan))
    # summaries
    summ = {}
    for mo in MODS:
        for grpname, sel in (("essential", lambda e: e < -0.5), ("nonessential", lambda e: e > -0.2)):
            row = {}
            for dd in DAYS:
                T = res["days"][dd][mo]["targets"]
                ts = [t for t in T if "essentiality" in T[t] and sel(T[t]["essentiality"])]
                med = lambda k: float(np.median([T[t][k] for t in ts if T[t].get(k) is not None])) if ts else None
                row[dd] = dict(n=len(ts), abundance=med("abundance_vs_nonessential"), abundance_ctrl=med("abundance_vs_control"),
                               residual=med("residual"), response=med("response"),
                               residual_ref=med("residual_ref"), response_ref=med("response_ref"))
            summ["%s_%s" % (mo, grpname)] = row
    res["summary"] = summ
    json.dump(res, open(outd + "/metrics_fate_drepanos.json", "w"), indent=1)
    for k, v in summ.items():
        print(k, json.dumps(v), flush=True)
    for mo in MODS:
        print("per-target", mo)
        T4 = res["days"][4][mo]["targets"]
        for t in sorted(T4, key=lambda t: T4[t].get("essentiality", 0)):
            if "essentiality" not in T4[t]:
                continue
            fmt = lambda v, f: (f % v) if v is not None else "-"
            print("  %-8s e=%5.2f" % (t, T4[t]["essentiality"]), " | ".join(
                "d%d a=%.2f r=%s R=%s" % (dd, res["days"][dd][mo]["targets"][t]["abundance_vs_nonessential"],
                                         fmt(res["days"][dd][mo]["targets"][t].get("residual_ref"), "%.2f"),
                                         fmt(res["days"][dd][mo]["targets"][t].get("response_ref"), "%.1f"))
                for dd in DAYS), flush=True)


if __name__ == "__main__":
    main()
