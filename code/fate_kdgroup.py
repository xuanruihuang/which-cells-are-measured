"""Per-cell knockdown groups in hESC vs K562 (A100; fate paper C6 core; predictions Q1-Q5 pre-stated in
notes/fate_paper_plan.md §3, commit d2d456d). Inputs: pan_extract_kdgroup.py output (hESC per target x guide parity x
group), pan_agg_guide_raw (only for the identical gene filter of fate_samescale.py), samescale2 K6 / K8 (K562 day 6 and
day 8 screens).
Genes, targets, effects and the K562 magnitude exactly as fate_samescale.py: genes with control >= 1 CP10K in hESC and
both K562 screens; log2 fold change of pseudobulk raw sums (+0.5) per library vs the controls; target gene zeroed; K562
magnitude = day-6 x day-8 cross-signal of all cells.
hESC magnitude per group (0 knocked down, 1 escaper, all = 0+1+2): guide parity 0 vs parity 1 of that group, each
against the non-targeting controls of the same parity. Groups need >= 10 cells in each parity.
Analysable targets: mean expected own count in controls (lambda x mean control library) >= 3.
Selection-bias null per target gene: non-targeting cells classified knocked-down-like on that gene, parity 0 vs 1, each
against all controls of the same parity.
Thinning null (added after the first run; commit 291ea33): null2 = control cells with the gene's count thinned by
Binomial(c, 0.2) and classified knocked-down-like; null3 = unthinned escaper-like control cells. Corrected magnitudes:
group effect vector minus the matching null effect vector, per parity, then the cross-signal.
Usage: fate_kdgroup.py <kdgroup dir> <pan_agg_guide_raw dir> <samescale2 dir> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr
from strata import essentiality
from fate_samescale import load_k562, lfc, MIN_CP10K

MIN_CELLS = 10
MIN_MU = 3.0


def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return dict(n=int(np.isfinite(a).sum()), n_both_positive=int(ok.sum()),
                ratio=float(2 ** np.median(np.log2(a[ok] / b[ok]))) if ok.any() else None)


def main():
    kd, pdir, kdir, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
    idx = json.load(open(kd + "/index.json")); pt, pg = idx["targets"], idx["genes"]; T, G = len(pt), len(pg)
    R = np.load(kd + "/raw_sums.npy", mmap_mode="r").reshape(T, 2, 3, G)
    C = np.load(kd + "/counts.npy").reshape(T, 2, 3, 4)
    NR = np.load(kd + "/null_raw.npy", mmap_mode="r"); NC = np.load(kd + "/null_cnt.npy")
    craw, clib = np.load(kd + "/ctrl_raw.npy"), np.load(kd + "/ctrl_lib.npy")
    nidx = {t: k for k, t in enumerate(idx["null_targets"])}
    N2R = np.load(kd + "/null2_raw.npy", mmap_mode="r"); N2C = np.load(kd + "/null2_cnt.npy")
    N3R = np.load(kd + "/null3_raw.npy", mmap_mode="r"); N3C = np.load(kd + "/null3_cnt.npy")
    # identical gene filter to fate_samescale.py
    pidx = json.load(open(pdir + "/index.json")); assert pidx["genes"] == pg
    praw = np.load(pdir + "/raw_sums.npy", mmap_mode="r").reshape(T, 4, G); pcnt = np.load(pdir + "/counts.npy").reshape(T, 4, 6)
    K6, K8 = load_k562(kdir + "/K6.npz"), load_k562(kdir + "/K8.npz")
    gset = sorted(set(K6["genes"]) & set(K8["genes"])); pgi = {g: j for j, g in enumerate(pg)}
    k6i, k8i = {g: j for j, g in enumerate(K6["genes"])}, {g: j for j, g in enumerate(K8["genes"])}
    ct = pt.index("non-targeting")
    hc = np.array([praw[ct, :, pgi[g]].sum() for g in gset]) / pcnt[ct, :, 4].sum() * 1e4
    c6 = np.array([K6["raw"][-1, :, k6i[g]].sum() for g in gset]) / K6["lib"][-1].sum() * 1e4
    c8 = np.array([K8["raw"][-1, :, k8i[g]].sum() for g in gset]) / K8["lib"][-1].sum() * 1e4
    keep = np.where((hc >= MIN_CP10K) & (c6 >= MIN_CP10K) & (c8 >= MIN_CP10K))[0]
    genes = [gset[j] for j in keep]; gq = {g: q for q, g in enumerate(genes)}
    hcol, c6col, c8col = (np.array([d[g] for g in genes]) for d in (pgi, k6i, k8i))
    shared = sorted(set(pt) & set(K6["targets"][:-1]) & set(K8["targets"][:-1]))
    lam = (praw[ct].sum(0) / pcnt[ct, :, 4].sum())                # per pan gene, all control cells
    mean_lib = idx["mean_control_library"]
    ess = essentiality()
    print("genes", len(genes), "shared", len(shared), flush=True)

    def zero(v, t):
        q = gq.get(t)
        if q is not None:
            v = v.copy(); v[q] = 0
        return v

    def hmag(ti, groups, t):
        u = []
        for p in (0, 1):
            n = C[ti, p, groups, 0].sum()
            if n < MIN_CELLS:
                return np.nan, 0
            rt = R[ti, p, groups][:, hcol].sum(0); lt = C[ti, p, groups, 1].sum()
            u.append(zero(lfc(rt, lt, craw[p, hcol], clib[p]), t))
        return float(u[0] @ u[1]), int(C[ti, :, groups, 0].sum())

    def hvec(ti, groups, t):
        out_ = []
        for p in (0, 1):
            if C[ti, p, groups, 0].sum() < MIN_CELLS:
                return None
            out_.append(zero(lfc(R[ti, p, groups][:, hcol].sum(0), C[ti, p, groups, 1].sum(), craw[p, hcol], clib[p]), t))
        return out_

    def nvec(NRx, NCx, k, t):
        if NCx[0, k, 0] < MIN_CELLS or NCx[1, k, 0] < MIN_CELLS:
            return None
        return [zero(lfc(NRx[p, k][hcol], NCx[p, k, 1], craw[p, hcol], clib[p]), t) for p in (0, 1)]

    def kmag(t):
        a = []
        for Kd, cols in ((K6, c6col), (K8, c8col)):
            k = Kd["targets"].index(t)
            a.append(zero(lfc(Kd["raw"][k][:, cols].sum(0), Kd["lib"][k].sum(), Kd["raw"][-1][:, cols].sum(0), Kd["lib"][-1].sum()), t))
        return float(a[0] @ a[1])

    rows = []
    for t in shared:
        if t not in pgi or lam[pgi[t]] * mean_lib < MIN_MU:
            continue
        ti = pt.index(t)
        m_kd, n_kd = hmag(ti, [0], t); m_esc, n_esc = hmag(ti, [1], t); m_all, n_all = hmag(ti, [0, 1, 2], t)
        n_und = int(C[ti, :, 2, 0].sum())
        resid = float(C[ti, :, :, 2].sum() / max(C[ti, :, :, 3].sum(), 1e-12))
        null = np.nan
        if t in nidx:
            k = nidx[t]
            if NC[0, k, 0] >= MIN_CELLS and NC[1, k, 0] >= MIN_CELLS:
                u = [zero(lfc(NR[p, k][hcol], NC[p, k, 1], craw[p, hcol], clib[p]), t) for p in (0, 1)]
                null = float(u[0] @ u[1])
        null2 = m_kd_corr = null3 = m_esc_corr = np.nan
        if t in nidx:
            k = nidx[t]; vk, b2 = hvec(ti, [0], t), nvec(N2R, N2C, k, t)
            if b2 is not None:
                null2 = float(b2[0] @ b2[1])
                if vk is not None:
                    m_kd_corr = float((vk[0] - b2[0]) @ (vk[1] - b2[1]))
            ve, b3 = hvec(ti, [1], t), nvec(N3R, N3C, k, t)
            if b3 is not None:
                null3 = float(b3[0] @ b3[1])
                if ve is not None:
                    m_esc_corr = float((ve[0] - b3[0]) @ (ve[1] - b3[1]))
        rows.append(dict(target=t, essentiality=float(ess.get(t, np.nan)), expected_ctrl=float(lam[pgi[t]] * mean_lib),
                         null2=null2, null3=null3, m_kd_corr=m_kd_corr, m_esc_corr=m_esc_corr,
                         resid=resid, n_kd=int(C[ti, :, 0, 0].sum()), n_esc=int(C[ti, :, 1, 0].sum()), n_und=n_und,
                         m_kd=m_kd, m_esc=m_esc, m_all=m_all, m_k562=kmag(t), null=null,
                         null_cells=[int(NC[p, nidx[t], 0]) for p in (0, 1)] if t in nidx else None))
    D = {k: np.array([r[k] for r in rows], dtype=float) for k in ("essentiality", "resid", "n_kd", "n_esc", "n_und", "m_kd", "m_esc", "m_all", "m_k562", "null",
                                                                    "null2", "null3", "m_kd_corr", "m_esc_corr")}
    e = D["essentiality"]; ess1 = np.isfinite(e) & (e < -1)
    escfrac = D["n_esc"] / np.maximum(D["n_kd"] + D["n_esc"], 1)
    res = {"rules": __doc__, "n_analysable": len(rows), "n_genes": len(genes)}
    # Q1
    ok1 = ess1 & np.isfinite(D["m_kd"]) & np.isfinite(D["m_esc"]) & (D["m_kd"] > 0)
    res["Q1_esc_over_kd_median_ess"] = med(D["m_esc"][ok1] / D["m_kd"][ok1]); res["Q1_n"] = int(ok1.sum())
    okall = np.isfinite(D["m_kd"]) & np.isfinite(D["m_esc"]) & (D["m_kd"] > 0)
    res["Q1b_esc_over_kd_median_all"] = med(D["m_esc"][okall] / D["m_kd"][okall]); res["Q1b_n"] = int(okall.sum())
    # Q2 / Q4
    dil = D["resid"] > 0.5
    res["Q2_kd_vs_k562_all"] = paired(D["m_kd"], D["m_k562"])
    res["Q2_kd_vs_k562_diluted"] = paired(D["m_kd"][dil], D["m_k562"][dil])
    res["Q2_kd_vs_k562_ess"] = paired(D["m_kd"][ess1], D["m_k562"][ess1])
    res["ref_all_cells_vs_k562_all"] = paired(D["m_all"], D["m_k562"])
    res["ref_all_cells_vs_k562_diluted"] = paired(D["m_all"][dil], D["m_k562"][dil])
    res["ref_all_cells_vs_k562_ess"] = paired(D["m_all"][ess1], D["m_k562"][ess1])
    # bh-style linear correction on the same targets (hESC side only; r_kd 0.2)
    eh = np.clip((D["resid"] - 0.2) / 0.8, 0, 0.95)
    res["Q4_linear_correction_all"] = paired(D["m_all"] / (1 - eh) ** 2, D["m_k562"])
    res["Q4_linear_correction_diluted"] = paired((D["m_all"] / (1 - eh) ** 2)[dil], D["m_k562"][dil])
    # Q3
    ok3 = np.isfinite(D["resid"]) & ((D["n_kd"] + D["n_esc"]) >= 20)
    res["Q3_spearman_escfrac_resid"] = float(spearmanr(escfrac[ok3], D["resid"][ok3])[0]); res["Q3_n"] = int(ok3.sum())
    res["undetermined_fraction_median"] = med(D["n_und"] / np.maximum(D["n_kd"] + D["n_esc"] + D["n_und"], 1))
    # Q5
    res["Q5_null_median_ess"] = med(D["null"][ess1]); res["Q5_mkd_median_ess"] = med(D["m_kd"][ess1])
    res["Q5_null_median_all"] = med(D["null"]); res["Q5_mkd_median_all"] = med(D["m_kd"])
    res["Q5_null_over_mkd_ess"] = (res["Q5_null_median_ess"] / res["Q5_mkd_median_ess"]) if res["Q5_mkd_median_ess"] else None
    # thinning null (Q5'), corrected magnitudes (Q2'), same-target comparison (Q4')
    res["Q5p_null2_median_ess"] = med(D["null2"][ess1]); res["Q5p_null2_median_all"] = med(D["null2"])
    res["Q5p_null2_over_mkd_ess"] = (res["Q5p_null2_median_ess"] / res["Q5_mkd_median_ess"]) if res["Q5_mkd_median_ess"] else None
    res["null3_median_ess"] = med(D["null3"][ess1])
    res["Q2p_kdcorr_vs_k562_all"] = paired(D["m_kd_corr"], D["m_k562"])
    res["Q2p_kdcorr_vs_k562_diluted"] = paired(D["m_kd_corr"][dil], D["m_k562"][dil])
    res["Q2p_kdcorr_vs_k562_ess"] = paired(D["m_kd_corr"][ess1], D["m_k562"][ess1])
    okc = ess1 & np.isfinite(D["m_kd_corr"]) & np.isfinite(D["m_esc_corr"]) & (D["m_kd_corr"] > 0)
    res["Q1p_esccorr_over_kdcorr_ess"] = med(D["m_esc_corr"][okc] / D["m_kd_corr"][okc]); res["Q1p_n"] = int(okc.sum())
    same = np.isfinite(D["m_kd_corr"]) & np.isfinite(D["m_all"])
    for nm, sel in (("all", same), ("diluted", same & dil), ("ess", same & ess1)):
        res["Q4p_same_targets_" + nm] = dict(n=int(sel.sum()), all_cells=paired(D["m_all"][sel], D["m_k562"][sel])["ratio"],
                                             linear=paired((D["m_all"] / (1 - eh) ** 2)[sel], D["m_k562"][sel])["ratio"],
                                             per_cell_raw=paired(D["m_kd"][sel], D["m_k562"][sel])["ratio"],
                                             per_cell_corrected=paired(D["m_kd_corr"][sel], D["m_k562"][sel])["ratio"])
    # tiers
    tiers = {}
    for nm, lo, hi in (("lt_-1", -np.inf, -1), ("-1_-0.5", -1, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf)):
        m = np.isfinite(e) & (e >= lo) & (e < hi)
        tiers[nm] = dict(n=int(m.sum()), resid=med(D["resid"][m]), esc_frac=med(escfrac[m]),
                         kd_vs_k562=paired(D["m_kd"][m], D["m_k562"][m])["ratio"], all_vs_k562=paired(D["m_all"][m], D["m_k562"][m])["ratio"],
                         esc_over_kd=med((D["m_esc"] / D["m_kd"])[m & okall]), null_median=med(D["null"][m]), mkd_median=med(D["m_kd"][m]),
                         null2_median=med(D["null2"][m]), kdcorr_vs_k562=paired(D["m_kd_corr"][m], D["m_k562"][m])["ratio"])
    res["tiers"] = tiers
    res["rows"] = rows
    json.dump(res, open(out + "/metrics_kdgroup.json", "w"), indent=1)
    for k, v in res.items():
        if k not in ("rules", "rows", "tiers"):
            print(k, json.dumps(v), flush=True)
    for k, v in tiers.items():
        print("tier", k, json.dumps(v), flush=True)


if __name__ == "__main__":
    main()
