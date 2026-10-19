"""hESC vs K562 on the same scale (A100; fate paper C4; expectations pre-stated in notes/fate_paper_plan.md §3,
commit 3074234). Pan et al. 2026: effects are weaker in hESC than in K562, also for well-knocked-down genes.
Inputs: pan_agg_guide_raw (hESC: per target x guide parity x random half: raw UMI sums, log1p sums, counts with
library and own-target raw), samescale2/{K6,K8}.npz (K562 day 6 essential and day 8 genome-wide screens: per target x
random half: raw UMI sums, log1p sums, cells, library, own-target raw).
Targets: shared by the Pan library and both K562 screens. Genes: control >= 1 CP10K in hESC and in both K562 screens
(pseudobulk). Target gene zeroed in its own effect.
Effects per unit:
  lfc    log2 fold change of pseudobulk (raw sums + 0.5) / library vs the controls of the same unit
  l1p    mean log1p CP10K minus the controls' mean (Pan-like scale)
Units: hESC guide halves (parity 0 / 1; independent transductions) and random halves; K562 independent screens (day 6
vs day 8, same guide element) and random halves within each screen.
Magnitude = cross-unit signal sum_g A . B (unbiased for the reproducible squared effect when A and B are independent).
Knockdown residual = own-target raw / (control rate x library).
Reported (E1-E3 of the plan): ratios hESC / K562 as medians of per-target log2 ratios on targets where both magnitudes
are positive (and the fraction kept), and as ratios of medians over all targets; overall, for matched good knockdown
(residual <= 0.3 in hESC and in both K562 screens), and per DepMap tier; hESC random / guide magnitude ratio (the
guide-specific part, which the K562 independent screens keep).
Usage: fate_samescale.py <pan_agg_guide_raw dir> <samescale2 dir> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
from strata import essentiality, TIERS

EPS = 0.5
KD = float(os.environ.get("FATE_KD", "0.3"))
MIN_CP10K = float(os.environ.get("FATE_MIN_CP10K", "1"))


def load_k562(path):
    z = np.load(path, allow_pickle=True)
    tl = [str(x) for x in z["targets"]]; K = len(tl)
    raw = z["raw"].astype(np.float64).reshape(K, 2, -1); l1p = z["l1p"].astype(np.float64).reshape(K, 2, -1)
    return dict(targets=tl, genes=[str(x) for x in z["genes"]], raw=raw, l1p=l1p, n=z["n"].reshape(K, 2).astype(float),
                lib=z["lib"].reshape(K, 2), own=z["own"].reshape(K, 2))


def lfc(rt, lt, rc, lc):
    return np.log2((rt + EPS) / lt) - np.log2((rc + EPS) / lc)


def summarize(mh, mk, sel):
    sel = sel & np.isfinite(mh) & np.isfinite(mk)          # targets with only one guide have no guide halves
    a, b = mh[sel], mk[sel]; pos = (a > 0) & (b > 0)
    return {"n": int(sel.sum()), "n_both_positive": int(pos.sum()),
            "paired_median_ratio": float(2 ** np.median(np.log2(a[pos] / b[pos]))) if pos.any() else None,
            "ratio_of_medians": float(np.median(a) / np.median(b)) if np.median(b) > 0 else None,
            "median_hesc": float(np.median(a)), "median_k562": float(np.median(b))}


def main():
    pdir, kdir, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
    idx = json.load(open(pdir + "/index.json")); pt, pg = idx["targets"], idx["genes"]; T, G = len(pt), len(pg)
    praw = np.load(pdir + "/raw_sums.npy", mmap_mode="r").reshape(T, 4, G)
    pl1p = np.load(pdir + "/sums64.npy", mmap_mode="r").reshape(T, 4, G)
    pcnt = np.load(pdir + "/counts.npy").reshape(T, 4, 6)
    pcraw, pclib = np.load(pdir + "/ctrl_raw.npy"), np.load(pdir + "/ctrl_lib.npy")
    K6, K8 = load_k562(kdir + "/K6.npz"), load_k562(kdir + "/K8.npz")
    gset = sorted(set(K6["genes"]) & set(K8["genes"])); pgi = {g: j for j, g in enumerate(pg)}
    k6i, k8i = {g: j for j, g in enumerate(K6["genes"])}, {g: j for j, g in enumerate(K8["genes"])}
    ct = pt.index("non-targeting")
    # controls (all codes / both halves) for the gene filter
    hc = np.array([praw[ct, :, pgi[g]].sum() for g in gset]) / pcnt[ct, :, 4].sum() * 1e4
    c6 = np.array([K6["raw"][-1, :, k6i[g]].sum() for g in gset]) / K6["lib"][-1].sum() * 1e4
    c8 = np.array([K8["raw"][-1, :, k8i[g]].sum() for g in gset]) / K8["lib"][-1].sum() * 1e4
    keep = np.where((hc >= MIN_CP10K) & (c6 >= MIN_CP10K) & (c8 >= MIN_CP10K))[0]
    genes = [gset[j] for j in keep]
    hcol, c6col, c8col = (np.array([d[g] for g in genes]) for d in (pgi, k6i, k8i))
    shared = sorted(set(pt) & set(K6["targets"][:-1]) & set(K8["targets"][:-1]))
    print("genes", len(genes), "shared targets", len(shared), flush=True)
    ess = essentiality(); tiers_e = np.array([ess.get(t, np.nan) for t in shared])
    gpos_sel = {g: q for q, g in enumerate(genes)}
    lam_h = pcraw.sum(0) / pclib.sum()
    res_h, res_6, res_8 = [], [], []
    mag = {k: [] for k in ("h_guide_lfc", "h_random_lfc", "h_guide_l1p", "h_random_l1p", "k_cross_lfc", "k_random6_lfc", "k_random8_lfc",
                           "k_cross_l1p", "k_random6_l1p", "k_random8_l1p")}
    def hunit(t, codes, kind):
        if kind == "lfc":
            rt = praw[t][codes][:, hcol].sum(0); lt = pcnt[t, codes, 4].sum()
            rc = praw[ct][codes][:, hcol].sum(0); lc = pcnt[ct, codes, 4].sum()
            return lfc(rt, lt, rc, lc)
        return pl1p[t][codes][:, hcol].sum(0) / pcnt[t, codes, 0].sum() - pl1p[ct][codes][:, hcol].sum(0) / pcnt[ct, codes, 0].sum()
    def kunit(Kd, cols, t, halves, kind):
        ti = Kd["targets"].index(t)
        if kind == "lfc":
            rt = Kd["raw"][ti, halves][:, cols].sum(0); lt = Kd["lib"][ti, halves].sum()
            rc = Kd["raw"][-1, halves][:, cols].sum(0); lc = Kd["lib"][-1, halves].sum()
            return lfc(rt, lt, rc, lc)
        return Kd["l1p"][ti, halves][:, cols].sum(0) / Kd["n"][ti, halves].sum() - Kd["l1p"][-1, halves][:, cols].sum(0) / Kd["n"][-1, halves].sum()
    for t in shared:
        ti = pt.index(t); z = gpos_sel.get(t, None)
        def zero(v):
            if z is not None:
                v = v.copy(); v[z] = 0
            return v
        for kind in ("lfc", "l1p"):
            gA, gB = zero(hunit(ti, [0, 1], kind)), zero(hunit(ti, [2, 3], kind))
            rA, rB = zero(hunit(ti, [0, 2], kind)), zero(hunit(ti, [1, 3], kind))
            mag["h_guide_" + kind].append(float(gA @ gB)); mag["h_random_" + kind].append(float(rA @ rB))
            a6, a8 = zero(kunit(K6, c6col, t, [0, 1], kind)), zero(kunit(K8, c8col, t, [0, 1], kind))
            mag["k_cross_" + kind].append(float(a6 @ a8))
            mag["k_random6_" + kind].append(float(zero(kunit(K6, c6col, t, [0], kind)) @ zero(kunit(K6, c6col, t, [1], kind))))
            mag["k_random8_" + kind].append(float(zero(kunit(K8, c8col, t, [0], kind)) @ zero(kunit(K8, c8col, t, [1], kind))))
        # knockdown residuals
        res_h.append(pcnt[ti, :, 5].sum() / max(lam_h[pgi[t]] * pcnt[ti, :, 4].sum(), 1e-12) if t in pgi else np.nan)
        for Kd, cols_i, out_l in ((K6, k6i, res_6), (K8, k8i, res_8)):
            k = Kd["targets"].index(t)
            if t in cols_i:
                lamk = Kd["raw"][-1, :, cols_i[t]].sum() / Kd["lib"][-1].sum()
                out_l.append(Kd["own"][k].sum() / max(lamk * Kd["lib"][k].sum(), 1e-12))
            else:
                out_l.append(np.nan)
    mag = {k: np.array(v) for k, v in mag.items()}
    res_h, res_6, res_8 = np.array(res_h), np.array(res_6), np.array(res_8)
    good = (res_h <= KD) & (res_6 <= KD) & (res_8 <= KD)
    allt = np.ones(len(shared), bool)
    depth = {"hesc_umi_per_cell": float(pcnt[:, :, 4].sum() / pcnt[:, :, 0].sum()),
             "k562_d6_umi_per_cell": float(K6["lib"].sum() / K6["n"].sum()), "k562_d8_umi_per_cell": float(K8["lib"].sum() / K8["n"].sum())}
    out_d = {"rules": __doc__, "kd_threshold": KD, "min_cp10k": MIN_CP10K, "n_genes": len(genes), "n_shared_targets": len(shared), "n_good_knockdown": int(good.sum()), "depth": depth,
             "residual_medians": {"hesc": float(np.nanmedian(res_h)), "k562_d6": float(np.nanmedian(res_6)), "k562_d8": float(np.nanmedian(res_8))}}
    comps = {"E1_l1p_random_all": ("h_random_l1p", "k_random8_l1p", allt),
             "E1b_l1p_random_all_vs_d6": ("h_random_l1p", "k_random6_l1p", allt),
             "E2_lfc_independent_goodkd": ("h_guide_lfc", "k_cross_lfc", good),
             "E2b_lfc_random_goodkd": ("h_random_lfc", "k_random8_lfc", good),
             "E2c_l1p_independent_goodkd": ("h_guide_l1p", "k_cross_l1p", good),
             "lfc_independent_all": ("h_guide_lfc", "k_cross_lfc", allt)}
    for name, (a, b, sel) in comps.items():
        out_d[name] = summarize(mag[a], mag[b], sel)
    tiers = {}
    for tn, lo, hi in TIERS:
        m = np.isfinite(tiers_e) & (tiers_e >= lo) & (tiers_e < hi)
        tiers[tn] = {"all_lfc_independent": summarize(mag["h_guide_lfc"], mag["k_cross_lfc"], m),
                     "goodkd_lfc_independent": summarize(mag["h_guide_lfc"], mag["k_cross_lfc"], m & good),
                     "residual_median_hesc": float(np.nanmedian(res_h[m])) if m.any() else None,
                     "residual_median_k562": float(np.nanmedian(np.r_[res_6[m], res_8[m]])) if m.any() else None}
    out_d["tiers"] = tiers
    # bootstrap over targets for the main comparison (E2)
    sel = good & np.isfinite(mag["h_guide_lfc"]) & (mag["h_guide_lfc"] > 0) & (mag["k_cross_lfc"] > 0)
    lr = np.log2(mag["h_guide_lfc"][sel] / mag["k_cross_lfc"][sel]); rng = np.random.default_rng(1)
    bs = [np.median(lr[rng.integers(0, len(lr), len(lr))]) for _ in range(1000)]
    out_d["E2_paired_ratio_ci95"] = [float(2 ** np.percentile(bs, 2.5)), float(2 ** np.percentile(bs, 97.5))]
    pos = (mag["h_guide_lfc"] > 0) & (mag["h_random_lfc"] > 0)
    out_d["hesc_random_over_guide_lfc_paired_median"] = float(2 ** np.median(np.log2(mag["h_random_lfc"][pos] / mag["h_guide_lfc"][pos])))
    pos = (mag["k_cross_lfc"] > 0) & (mag["k_random8_lfc"] > 0)
    out_d["k562_random8_over_cross_lfc_paired_median"] = float(2 ** np.median(np.log2(mag["k_random8_lfc"][pos] / mag["k_cross_lfc"][pos])))
    np.savez_compressed(out + "/per_target.npz", targets=np.array(shared), essentiality=tiers_e, res_h=res_h, res_6=res_6, res_8=res_8, **mag)
    json.dump(out_d, open(out + "/metrics_fate_samescale.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out_d.items() if k not in ("rules", "tiers")}, indent=0), flush=True)
    for k, v in tiers.items():
        print(k, json.dumps(v), flush=True)


if __name__ == "__main__":
    main()
