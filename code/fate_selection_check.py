"""R3 (fate paper §8; expectations S1, S3-S5 pre-stated in notes/fate_paper_plan.md §8): is the matched hESC-K562
comparison (residual <= KD on both sides; ratio 1.18) a selection on the hESC outcome, and how does the ratio depend on
hESC depletion, on the estimator (all cells vs silenced cells) and on the K562 harvest day?
Inputs: fate_samescale.py per_target.npz (1,328 shared targets: residuals, magnitudes as independent units and random
halves); pan_library_norm.py rows (per target: library-input-normalised abundance 'norm', oligo copies per guide = Pan's
expected dropout); fate_kdgroup.py rows (per target: silenced-cell magnitude m_kd_corr, all-cell m_all, K562 m_k562).
A. composition of the matched set vs all, by DepMap tier, abundance bin and copies bin; B. ratio by abundance bin under
four estimators; C. direct standardisation of the matched set to the full tier x abundance distribution; D. ratio against
K562 day 6 only and day 8 only (random halves both sides); E. residual medians on both sides.
Usage: fate_selection_check.py <samescale_out> <pan_libnorm json> <kdgroup json> <out json>. Exploratory."""
import sys, json
import numpy as np

TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf))
ABINS = (("<=0.05", -np.inf, 0.05), ("0.05-0.2", 0.05, 0.2), ("0.2-0.5", 0.2, 0.5), (">0.5", 0.5, np.inf))
CBINS = (("1", 0, 1.5), ("2-3", 1.5, 3.5), ("4-7", 3.5, 7.5), ("8-15", 7.5, 15.5), (">=16", 15.5, np.inf))
KD = 0.3
NBOOT = 1000


def paired(a, b, rng, w=None):
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if w is not None:
        ok &= np.isfinite(w) & (w > 0)
    if ok.sum() < 5:
        return dict(n=int(ok.sum()), ratio=None, ci95=None)
    lr = np.log2(a[ok] / b[ok]); n = len(lr)
    if w is None:
        bs = [np.median(lr[rng.integers(0, n, n)]) for _ in range(NBOOT)]
        return dict(n=int(n), ratio=float(2 ** np.median(lr)), ci95=[float(2 ** np.percentile(bs, 2.5)), float(2 ** np.percentile(bs, 97.5))])
    ww = w[ok]
    def wmed(x, wt):
        o = np.argsort(x); c = np.cumsum(wt[o]); return x[o][np.searchsorted(c, c[-1] / 2)]
    bs = []
    for _ in range(NBOOT):
        i = rng.integers(0, n, n); bs.append(wmed(lr[i], ww[i]))
    return dict(n=int(n), ratio=float(2 ** wmed(lr, ww)), ci95=[float(2 ** np.percentile(bs, 2.5)), float(2 ** np.percentile(bs, 97.5))])


def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1]), int(ok.sum())


def binof(x, bins):
    out = np.array([None] * len(x), dtype=object)
    for nm, lo, hi in bins:
        out[(x > lo) & (x <= hi)] = nm
    return out


def main():
    sdir, ljson, kjson, outf = sys.argv[1:5]; rng = np.random.default_rng(1)
    z = np.load(sdir + "/per_target.npz", allow_pickle=True); tl = [str(t) for t in z["targets"]]; T = len(tl)
    ess, rh, r6, r8 = z["essentiality"], z["res_h"], z["res_6"], z["res_8"]
    hg, kc, hr, k6, k8 = z["h_guide_lfc"], z["k_cross_lfc"], z["h_random_lfc"], z["k_random6_lfc"], z["k_random8_lfc"]
    L = {r["target"]: r for r in json.load(open(ljson))["rows"]}
    norm = np.array([L[t]["norm"] if t in L else np.nan for t in tl]); copies = np.array([L[t]["copies_per_guide"] if t in L else np.nan for t in tl])
    K = {r["target"]: r for r in json.load(open(kjson))["rows"]}
    mkd = np.array([K[t]["m_kd_corr"] if t in K else np.nan for t in tl]); mkall = np.array([K[t]["m_all"] if t in K else np.nan for t in tl])
    mk562 = np.array([K[t]["m_k562"] if t in K else np.nan for t in tl]); nkd = np.array([K[t]["n_kd"] if t in K else np.nan for t in tl])
    matched = (rh <= KD) & (r6 <= KD) & (r8 <= KD)
    k_only = (r6 <= KD) & (r8 <= KD)
    tier = binof(ess, TIERS); abin = binof(norm, ABINS); cbin = binof(copies, CBINS)
    res = {"rules": __doc__, "n_shared": T, "n_matched": int(matched.sum()), "n_k562_only_matched": int(k_only.sum()),
           "n_with_norm": int(np.isfinite(norm).sum()), "n_classifiable": int(np.isfinite(mkd).sum())}
    # A. composition
    A = {"by_tier": {}, "by_abundance": {}, "by_copies": {}}
    for nm, _, _ in TIERS:
        m = tier == nm
        A["by_tier"][nm] = dict(n_all=int(m.sum()), n_matched=int((m & matched).sum()), frac_retained=float((m & matched).sum() / max(m.sum(), 1)),
                                norm_median_matched=med(norm[m & matched]), norm_median_unmatched=med(norm[m & ~matched]),
                                copies_median_matched=med(copies[m & matched]), copies_median_unmatched=med(copies[m & ~matched]),
                                resid_h_median=med(rh[m]), resid_k_median=med(np.r_[r6[m], r8[m]]))
    for nm, _, _ in ABINS:
        m = abin == nm
        A["by_abundance"][nm] = dict(n_all=int(m.sum()), n_matched=int((m & matched).sum()), frac_retained=float((m & matched).sum() / max(m.sum(), 1)),
                                     share_all=float(m.sum() / np.isfinite(norm).sum()), share_matched=float((m & matched).sum() / max((matched & np.isfinite(norm)).sum(), 1)))
    for nm, _, _ in CBINS:
        m = cbin == nm
        A["by_copies"][nm] = dict(n_all=int(m.sum()), n_matched=int((m & matched).sum()), frac_retained=float((m & matched).sum() / max(m.sum(), 1)))
    A["spearman_resid_h_vs_norm"] = spearman(rh, norm); A["spearman_resid_h_vs_norm_most_essential"] = spearman(rh[tier == "lt_-1"], norm[tier == "lt_-1"])
    A["spearman_resid_h_vs_copies"] = spearman(rh, copies)
    res["A_composition"] = A
    # B. ratio by abundance bin under four estimators
    B = {}
    for nm, _, _ in ABINS + (("all", -np.inf, np.inf),):
        m = np.ones(T, bool) if nm == "all" else (abin == nm)
        B[nm] = dict(all_cells_independent=paired(hg[m], kc[m], rng), matched_both=paired(hg[m & matched], kc[m & matched], rng),
                     k562_only_matched=paired(hg[m & k_only], kc[m & k_only], rng),
                     silenced_cells=paired(mkd[m], mk562[m], rng), all_cells_same_targets=paired(mkall[m & np.isfinite(mkd)], mk562[m & np.isfinite(mkd)], rng),
                     silenced_cells_most_essential=paired(mkd[m & (tier == "lt_-1")], mk562[m & (tier == "lt_-1")], rng),
                     n_kd_cells_median=med(nkd[m & np.isfinite(mkd)]), resid_h_median=med(rh[m]))
    res["B_by_abundance"] = B
    # per tier, matched vs all (reference) and silenced
    res["B_by_tier"] = {nm: dict(all_cells=paired(hg[tier == nm], kc[tier == nm], rng), matched=paired(hg[(tier == nm) & matched], kc[(tier == nm) & matched], rng),
                                 silenced=paired(mkd[tier == nm], mk562[tier == nm], rng)) for nm, _, _ in TIERS}
    # C. direct standardisation of the matched set to the full tier x abundance distribution
    cell = np.array(["%s|%s" % (a, b) for a, b in zip(tier, abin)], dtype=object)
    have = np.array([t is not None and a is not None for t, a in zip(tier, abin)])
    full_share = {c: float((cell[have] == c).mean()) for c in set(cell[have])}
    mm = matched & have & np.isfinite(hg) & np.isfinite(kc) & (hg > 0) & (kc > 0)
    matched_share = {c: float((cell[mm] == c).mean()) for c in set(cell[mm])}
    w = np.array([full_share[c] / matched_share[c] if (mm[i] and c in matched_share) else np.nan for i, c in enumerate(cell)])
    covered = sum(full_share[c] for c in matched_share)
    res["C_standardised"] = dict(weighted_ratio=paired(hg, kc, rng, w=w), coverage_of_full_weight=float(covered),
                                 cells_without_matched_targets={c: full_share[c] for c in full_share if c not in matched_share})
    # D. harvest day: K562 day 6 only and day 8 only, random halves on both sides
    D = {}
    for sname, m in (("all", np.ones(T, bool)), ("matched", matched), ("most_essential_matched", matched & (tier == "lt_-1")), ("most_essential_all", tier == "lt_-1")):
        D[sname] = dict(vs_day6_random=paired(hr[m], k6[m], rng), vs_day8_random=paired(hr[m], k8[m], rng), vs_cross_guide=paired(hg[m], kc[m], rng),
                        resid_median_hesc=med(rh[m]), resid_median_d6=med(r6[m]), resid_median_d8=med(r8[m]))
    res["D_harvest_day"] = D
    res["E_matched_residuals"] = dict(hesc=med(rh[matched]), k562_d6=med(r6[matched]), k562_d8=med(r8[matched]),
                                      hesc_all=med(rh), k562_d6_all=med(r6), k562_d8_all=med(r8))
    json.dump(res, open(outf, "w"), indent=1)
    for k in ("A_composition", "B_by_abundance", "B_by_tier", "C_standardised", "D_harvest_day", "E_matched_residuals"):
        print("==", k); print(json.dumps(res[k], indent=None)[:6000], flush=True)


if __name__ == "__main__":
    main()
