"""Fate v3r2 external review, checks V1, V2, V9, V16 (expectations pre-stated in notes/fate_analysis_redesign.md §7.2,
commit f9d9a81). A100.
V1  symmetric silenced-cell comparison hESC / K562. Genes: those of fate_samescale.py (control >= 1 CP10K in hESC and
    both K562 screens) that are also kept by both K562 budget extractions. hESC (pan_agg_kdgroup): all cells and
    knocked-down cells minus the thinning null (null2), per guide parity, cross-signal between parities (as
    fate_kdgroup.py m_all / m_kd_corr). K562 (fate_budget_extract.py outputs, day-6 essential and day-8 genome-scale):
    per screen both random halves pooled, all cells and knocked-down cells minus that screen's thinning null (as
    fate_budget_screens.py), cross-signal between screens. Ratios on the targets where all four magnitudes exist, by
    hESC library-normalised abundance bin, DepMap stratum, the diluted subset (hESC residual > 0.5) and all.
V2  estimators that keep non-positive magnitudes: ratio of medians (primary), ratio of sums, and the median of per-target
    log ratios over targets positive on both sides (former estimator); paired bootstrap over targets, 2,000 draws;
    counts of non-positive magnitudes per side. Applied to fate_samescale per_target.npz (guide halves vs K562 screens;
    and random halves vs each K562 screen) and to V1.
V9  cell-number shrinkage: hESC targets with library-normalised abundance > 0.5 have their summed counts thinned
    binomially to the median cell number per parity of the <= 0.05 bin; magnitude and ratio recomputed. Pseudocount
    0.05 instead of 0.5 for all targets and both contexts.
V16 the nine threshold combinations of fate_samescale.py (samescale_rob).
Usage: fate_rev_samescale.py <out_dir>. Exploratory."""
import os, sys, json, glob
import numpy as np
import config
from fate_budget import Ctx, essentiality
from fate_budget_screens import pooled, pooled_null
from fate_samescale import load_k562

FI = config.get("FATE_ROOT")
DEPMAP = config.get("DEPMAP")
KDG, PGR, SS2 = config.get("PAN_KDGROUP"), config.get("PAN_GUIDE_RAW"), config.get("FATE_ROOT") + "/samescale2"
MIN_CP10K, MIN_CELLS, MIN_MU, NB = 1.0, 10, 3.0, 2000
TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf))
ABINS = (("<=0.05", -np.inf, 0.05), ("0.05-0.2", 0.05, 0.2), ("0.2-0.5", 0.2, 0.5), (">0.5", 0.5, np.inf))


def lfc(rt, lt, rc, lc, eps=0.5):
    return np.log2((rt + eps) / lt) - np.log2((rc + eps) / lc)


def est(a, b, rng, nb=NB):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]; n = len(a)
    out = dict(n=int(n), n_nonpos_hesc=int((a <= 0).sum()), n_nonpos_k562=int((b <= 0).sum()))
    if n < 5:
        return out
    pos = (a > 0) & (b > 0)

    def stats(ix):
        aa, bb = a[ix], b[ix]; pp = (aa > 0) & (bb > 0)
        rm = np.median(aa) / np.median(bb) if np.median(bb) > 0 else np.nan
        rs = aa.sum() / bb.sum() if bb.sum() > 0 else np.nan
        pm = 2 ** np.median(np.log2(aa[pp] / bb[pp])) if pp.sum() >= 3 else np.nan
        return rm, rs, pm
    pt = stats(np.arange(n)); B = np.array([stats(rng.integers(0, n, n)) for _ in range(nb)])
    ci = lambda k: [float(np.nanpercentile(B[:, k], 2.5)), float(np.nanpercentile(B[:, k], 97.5))]
    out.update(ratio_of_medians=float(pt[0]), ratio_of_medians_ci=ci(0), ratio_of_sums=float(pt[1]), ratio_of_sums_ci=ci(1),
               paired_median_positive=float(pt[2]), paired_median_positive_ci=ci(2), n_both_positive=int(pos.sum()),
               median_hesc=float(np.median(a)), median_k562=float(np.median(b)))
    return out


def binof(x, bins):
    out = np.array([None] * len(x), dtype=object)
    for nm, lo, hi in bins:
        out[(x > lo) & (x <= hi)] = nm
    return out


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); rng = np.random.default_rng(7); res = {"rules": __doc__}
    ess = essentiality(DEPMAP)
    L = {r["target"]: r for r in json.load(open(FI + "/pan_libnorm/pan_library_norm.json"))["rows"]}
    # ---------------- V2 on the existing per-target table
    z = np.load(FI + "/samescale_out/per_target.npz", allow_pickle=True); tl = [str(t) for t in z["targets"]]
    E = z["essentiality"]; rh, r6, r8 = z["res_h"], z["res_6"], z["res_8"]
    hg, kc, hr, k6, k8 = z["h_guide_lfc"], z["k_cross_lfc"], z["h_random_lfc"], z["k_random6_lfc"], z["k_random8_lfc"]
    matched = (rh <= 0.3) & (r6 <= 0.3) & (r8 <= 0.3)
    norm = np.array([L[t]["norm"] if t in L and L[t]["norm"] is not None else np.nan for t in tl], float)
    tier = binof(-E, tuple((nm, -hi, -lo) for nm, lo, hi in TIERS)); abin = binof(norm, ABINS)
    V2 = {}
    sets = [("all", np.ones(len(tl), bool)), ("matched", matched)] + [("tier_" + nm, tier == nm) for nm, _, _ in TIERS] + \
           [("tier_%s_matched" % nm, (tier == nm) & matched) for nm, _, _ in TIERS] + [("abundance_" + nm, abin == nm) for nm, _, _ in ABINS]
    for nm, m in sets:
        V2[nm] = dict(guide_vs_cross=est(hg[m], kc[m], rng), random_vs_d6=est(hr[m], k6[m], rng), random_vs_d8=est(hr[m], k8[m], rng))
        print("V2", nm, json.dumps(V2[nm]["guide_vs_cross"]), flush=True)
    res["V2"] = V2
    # ---------------- V16
    V16 = {}
    for f in sorted(glob.glob(FI + "/samescale_rob/*/metrics_fate_samescale.json")):
        d = json.load(open(f)); V16[f.split("/")[-2]] = dict(n_good=d["n_good_knockdown"], E2=d["E2_lfc_independent_goodkd"], E2_ci=d.get("E2_paired_ratio_ci95"),
                                                           all=d["lfc_independent_all"], n_genes=d["n_genes"])
    res["V16"] = V16
    print("V16", json.dumps({k: (v["E2"]["paired_median_ratio"], v["E2"]["ratio_of_medians"], v["n_good"]) for k, v in V16.items()}), flush=True)
    # ---------------- V1
    idx = json.load(open(KDG + "/index.json")); pt, pg = idx["targets"], idx["genes"]; T, G = len(pt), len(pg)
    R = np.load(KDG + "/raw_sums.npy", mmap_mode="r").reshape(T, 2, 3, G); C = np.load(KDG + "/counts.npy").reshape(T, 2, 3, 4)
    craw, clib = np.load(KDG + "/ctrl_raw.npy"), np.load(KDG + "/ctrl_lib.npy")
    nidx = {t: k for k, t in enumerate(idx["null_targets"])}
    N2R = np.load(KDG + "/null2_raw.npy", mmap_mode="r"); N2C = np.load(KDG + "/null2_cnt.npy")
    pidx = json.load(open(PGR + "/index.json")); assert pidx["genes"] == pg
    praw = np.load(PGR + "/raw_sums.npy", mmap_mode="r").reshape(T, 4, G); pcnt = np.load(PGR + "/counts.npy").reshape(T, 4, 6)
    K6, K8 = load_k562(SS2 + "/K6.npz"), load_k562(SS2 + "/K8.npz")
    gset = sorted(set(K6["genes"]) & set(K8["genes"])); pgi = {g: j for j, g in enumerate(pg)}
    k6i, k8i = {g: j for j, g in enumerate(K6["genes"])}, {g: j for j, g in enumerate(K8["genes"])}
    ct = pt.index("non-targeting")
    hc = np.array([praw[ct, :, pgi[g]].sum() for g in gset]) / pcnt[ct, :, 4].sum() * 1e4
    c6 = np.array([K6["raw"][-1, :, k6i[g]].sum() for g in gset]) / K6["lib"][-1].sum() * 1e4
    c8 = np.array([K8["raw"][-1, :, k8i[g]].sum() for g in gset]) / K8["lib"][-1].sum() * 1e4
    genes_ss = [gset[j] for j in np.where((hc >= MIN_CP10K) & (c6 >= MIN_CP10K) & (c8 >= MIN_CP10K))[0]]
    A, B = Ctx(FI + "/budget", "K562"), Ctx(FI + "/budget_d8", "K562d8")
    genes = [g for g in genes_ss if g in A.gq and g in B.gq]; gq = {g: q for q, g in enumerate(genes)}
    hcol = np.array([pgi[g] for g in genes]); selA = np.array([A.gq[g] for g in genes]); selB = np.array([B.gq[g] for g in genes])
    print("V1 genes: samescale", len(genes_ss), "kept", len(genes), flush=True)
    lam = praw[ct].sum(0) / pcnt[ct, :, 4].sum(); mean_lib = idx["mean_control_library"]
    tA = {t: k for k, t in enumerate(A.tl)}; tB = {t: k for k, t in enumerate(B.tl)}

    def zero(v, t):
        q = gq.get(t)
        if q is not None:
            v = v.copy(); v[q] = 0
        return v

    def hvec(ti, groups, t):
        o = []
        for p in (0, 1):
            if C[ti, p, groups, 0].sum() < MIN_CELLS:
                return None
            o.append(zero(lfc(R[ti, p, groups][:, hcol].sum(0), C[ti, p, groups, 1].sum(), craw[p, hcol], clib[p]), t))
        return o

    rows = []
    for t in tl:
        if t not in pgi or lam[pgi[t]] * mean_lib < MIN_MU or t not in tA or t not in tB or t not in nidx:
            continue
        ti = pt.index(t); k = nidx[t]
        va, vk = hvec(ti, [0, 1, 2], t), hvec(ti, [0], t)
        if N2C[0, k, 0] < MIN_CELLS or N2C[1, k, 0] < MIN_CELLS or va is None:
            continue
        nl = [zero(lfc(N2R[p, k][hcol], N2C[p, k, 1], craw[p, hcol], clib[p]), t) for p in (0, 1)]
        h_all = float(va[0] @ va[1]); h_kd = float((vk[0] - nl[0]) @ (vk[1] - nl[1])) if vk is not None else np.nan
        ka, kb = pooled(A, tA[t], [0, 1, 2], selA), pooled(B, tB[t], [0, 1, 2], selB)
        if ka is None or kb is None:
            continue
        k_all = float(zero(ka[0], t) @ zero(kb[0], t))
        kda, kdb = pooled(A, tA[t], [0], selA), pooled(B, tB[t], [0], selB)
        nla, nlb = pooled_null(A, tA[t], selA), pooled_null(B, tB[t], selB)
        k_kd = float(zero(kda[0] - nla, t) @ zero(kdb[0] - nlb, t)) if all(x is not None for x in (kda, kdb, nla, nlb)) else np.nan
        rows.append(dict(target=t, essentiality=float(ess.get(t, np.nan)), norm=L[t]["norm"] if t in L else None,
                         resid_h=float(C[ti, :, :, 2].sum() / max(C[ti, :, :, 3].sum(), 1e-12)),
                         h_all=h_all, h_kd=h_kd, k_all=k_all, k_kd=k_kd,
                         n_kd_h=int(C[ti, :, 0, 0].sum()), n_kd_k=[None if kda is None else kda[2], None if kdb is None else kdb[2]]))
    res["V1_rows"] = rows
    D = {k: np.array([np.nan if r[k] is None else r[k] for r in rows], float) for k in ("essentiality", "norm", "resid_h", "h_all", "h_kd", "k_all", "k_kd")}
    full = np.isfinite(D["h_all"]) & np.isfinite(D["h_kd"]) & np.isfinite(D["k_all"]) & np.isfinite(D["k_kd"])
    t1 = binof(-D["essentiality"], tuple((nm, -hi, -lo) for nm, lo, hi in TIERS)); a1 = binof(D["norm"], ABINS)
    # consistency of the budget-derived K562 all-cell magnitude with the samescale one
    kcd = dict(zip(tl, kc)); kref = np.array([kcd.get(r["target"], np.nan) for r in rows])
    okc = np.isfinite(kref) & np.isfinite(D["k_all"])
    res["V1_check_k_all_vs_samescale"] = dict(n=int(okc.sum()), pearson=float(np.corrcoef(kref[okc], D["k_all"][okc])[0, 1]),
                                              median_ratio=float(np.median(D["k_all"][okc & (kref > 0) & (D["k_all"] > 0)] / kref[okc & (kref > 0) & (D["k_all"] > 0)])))
    print("V1 check", json.dumps(res["V1_check_k_all_vs_samescale"]), flush=True)
    V1 = {}
    sets = [("all_classifiable", full), ("diluted_resid_gt_0.5", full & (D["resid_h"] > 0.5))] + \
           [("tier_" + nm, full & (t1 == nm)) for nm, _, _ in TIERS] + [("abundance_" + nm, full & (a1 == nm)) for nm, _, _ in ABINS] + \
           [("abundance_<=0.05_most_essential", full & (a1 == "<=0.05") & (t1 == "lt_-1"))]
    for nm, m in sets:
        kk = D["k_kd"][m] / D["k_all"][m]; kpos = (D["k_all"][m] > 0) & (D["k_kd"][m] > 0)
        hh = (D["h_all"][m] > 0) & (D["h_kd"][m] > 0)
        V1[nm] = dict(n=int(m.sum()), all_cells=est(D["h_all"][m], D["k_all"][m], rng), asymmetric_silenced=est(D["h_kd"][m], D["k_all"][m], rng),
                      symmetric_silenced=est(D["h_kd"][m], D["k_kd"][m], rng),
                      k562_selection_term_median=float(2 ** np.median(np.log2(kk[kpos]))) if kpos.sum() else None,
                      hesc_selection_term_median=float(2 ** np.median(np.log2((D["h_kd"][m] / D["h_all"][m])[hh]))) if hh.sum() else None)
        a_, s_ = V1[nm]["all_cells"].get("paired_median_positive"), V1[nm]["symmetric_silenced"].get("paired_median_positive")
        if a_ and s_ and a_ < 1 and a_ > 0 and s_ > 0:
            V1[nm]["log_gap_closed_paired"] = float(1 - np.log2(min(s_, 1)) / np.log2(a_))
        a_, s_ = V1[nm]["all_cells"].get("ratio_of_medians"), V1[nm]["symmetric_silenced"].get("ratio_of_medians")
        if a_ and s_ and 0 < a_ < 1 and s_ > 0:
            V1[nm]["log_gap_closed_ratio_of_medians"] = float(1 - np.log2(min(s_, 1)) / np.log2(a_))
        print("V1", nm, json.dumps(V1[nm]), flush=True)
    res["V1"] = V1
    # ---------------- V9
    cpt = pcnt[:, :, 0].reshape(T, 2, 2).sum(2)                                  # cells per (target, parity)
    sidx = [pt.index(t) for t in tl]; normv = norm
    lowbin = np.array([(normv[i] <= 0.05) if np.isfinite(normv[i]) else False for i in range(len(tl))])
    n_ref = np.median(cpt[np.array(sidx)[lowbin]], axis=0)
    hcol_ss = np.array([pgi[g] for g in genes_ss]); gq_ss = {g: q for q, g in enumerate(genes_ss)}
    rc = [praw[ct][[0, 1]][:, hcol_ss].sum(0), praw[ct][[2, 3]][:, hcol_ss].sum(0)]; lc = [pcnt[ct, [0, 1], 4].sum(), pcnt[ct, [2, 3], 4].sum()]

    def hmag(ti, t, f=(1.0, 1.0), eps=0.5, rngx=None):
        v = []
        for p, codes in enumerate(([0, 1], [2, 3])):
            rt = praw[ti][codes][:, hcol_ss].sum(0); lt = pcnt[ti, codes, 4].sum()
            if f[p] < 1:
                rt = rngx.binomial(rt.astype(np.int64), f[p]).astype(float); lt = lt * f[p]
            x = lfc(rt, lt, rc[p], lc[p], eps); q = gq_ss.get(t)
            if q is not None:
                x[q] = 0
            v.append(x)
        return float(v[0] @ v[1])
    hi = np.array([(normv[i] > 0.5) if np.isfinite(normv[i]) else False for i in range(len(tl))])
    orig, thin, kk = [], [], []
    rr = np.random.default_rng(11)
    for i in np.where(hi)[0]:
        ti = sidx[i]; f = tuple(min(1.0, n_ref[p] / max(cpt[ti, p], 1)) for p in (0, 1))
        orig.append(hmag(ti, tl[i])); thin.append(np.median([hmag(ti, tl[i], f, rngx=rr) for _ in range(3)])); kk.append(kc[i])
    orig, thin, kk = np.array(orig), np.array(thin), np.array(kk)
    res["V9_thinning"] = dict(n_ref_cells_per_parity=[float(x) for x in n_ref], n=int(len(orig)), original=est(orig, kk, rng), thinned=est(thin, kk, rng),
                              median_thinned_over_original=float(np.median(thin[(orig > 0) & (thin > 0)] / orig[(orig > 0) & (thin > 0)])))
    print("V9 thinning", json.dumps(res["V9_thinning"]), flush=True)
    # pseudocount 0.05, both contexts, all shared targets
    c6col, c8col = np.array([k6i[g] for g in genes_ss]), np.array([k8i[g] for g in genes_ss])

    def kmag(t, eps):
        a = []
        for Kd, cols in ((K6, c6col), (K8, c8col)):
            k = Kd["targets"].index(t)
            x = lfc(Kd["raw"][k][:, cols].sum(0), Kd["lib"][k].sum(), Kd["raw"][-1][:, cols].sum(0), Kd["lib"][-1].sum(), eps); q = gq_ss.get(t)
            if q is not None:
                x[q] = 0
            a.append(x)
        return float(a[0] @ a[1])
    h05, k05 = np.array([hmag(sidx[i], tl[i], eps=0.05) for i in range(len(tl))]), np.array([kmag(t, 0.05) for t in tl])
    res["V9_pseudocount"] = {nm: dict(eps_0_5=est(hg[m], kc[m], rng, 500), eps_0_05=est(h05[m], k05[m], rng, 500))
                             for nm, m in (("all", np.ones(len(tl), bool)), ("matched", matched), ("abundance_<=0.05", abin == "<=0.05"))}
    print("V9 pseudocount", json.dumps(res["V9_pseudocount"]), flush=True)
    json.dump(res, open(out + "/metrics_rev_samescale.json", "w"), indent=1, default=lambda v: None if v is None else float(v))


if __name__ == "__main__":
    main()
