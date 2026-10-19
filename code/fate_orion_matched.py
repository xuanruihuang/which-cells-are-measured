"""R1 (fate paper §8; expectations O1-O4 pre-stated in notes/fate_paper_plan.md §8, commit a42ae7f): does the published
HCT116 > HEK293T difference in X-Atlas/Orion (discernible-response fraction 48.4% vs 40.4%; median knockdown 75.4% vs
51.5%) survive pairing targets and matching knockdown on both sides?
Input: orion_extract.py aggregate (orion_agg), same layout as fate_orion_c1.py:
  counts.npy  rows (line * T + target) * 2 + half, cols n, G1, S, G2M, lib_sum, own_target_raw_sum
  sums64.npy  same rows x genes: sum of log1p CP10K over the group's cells
  ctrl_raw.npy / ctrl_lib.npy  per line: control raw-count sums per gene, control library sum
Per line and target (>= MIN_CELLS cells, >= 5 per half): response = random-half cross-signal of (mean log1p of the half
minus the control's same half) over the response genes = control >= MIN_RESP CP10K in BOTH lines, target gene zeroed;
residual = own raw / (control rate x library), pooled and per half. Inert targets: own gene < 0.01 CP10K in that line's
controls. Discernible: response above the 97.5th percentile of the inert targets of the same line.
Comparisons on shared targets: paired median log2 ratio HEK293T / HCT116 (both positive), all targets vs residual <=
KD in both lines, over a sweep of KD and MIN_RESP; per DepMap tier; discernible fraction per line, all vs matched.
Premise: residual correlation between halves across targets >= 0.5 in both lines.
Usage: fate_orion_matched.py <orion_agg dir> <depmap csv> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np

TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf))
MIN_CELLS = 20
NBOOT = 1000


def essentiality(path):
    with open(path) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        rows = [np.array([float(x) if x else np.nan for x in line.rstrip("\n").split(",")[1:]]) for line in f]
    return dict(zip(sym, np.nanmedian(np.array(rows), 0)))


def paired_ratio(a, b, rng):
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if ok.sum() < 5:
        return dict(n=int(ok.sum()), ratio=None, ci95=None)
    lr = np.log2(a[ok] / b[ok]); n = len(lr)
    bs = np.array([np.median(lr[rng.integers(0, n, n)]) for _ in range(NBOOT)])
    return dict(n=int(n), ratio=float(2 ** np.median(lr)), ci95=[float(2 ** np.percentile(bs, 2.5)), float(2 ** np.percentile(bs, 97.5))])


def frac_ci(x, rng):
    x = np.asarray(x, bool); n = len(x)
    if n < 5:
        return dict(n=int(n), frac=None, ci95=None)
    bs = np.array([x[rng.integers(0, n, n)].mean() for _ in range(NBOOT)])
    return dict(n=int(n), frac=float(x.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def main():
    d, dep, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True); ess = essentiality(dep)
    idx = json.load(open(d + "/index.json")); lines, targets, genes = idx["lines"], idx["targets"], idx["genes"]
    T = len(targets); gpos = {g: j for j, g in enumerate(genes)}; ct = targets.index(idx["ctrl"])
    cnt = np.load(d + "/counts.npy"); S = np.load(d + "/sums64.npy", mmap_mode="r")
    craw = np.load(d + "/ctrl_raw.npy"); clib = np.load(d + "/ctrl_lib.npy")
    rate = [craw[li] / clib[li] for li in range(len(lines))]; cp = [r * 1e4 for r in rate]
    row = lambda li, t, h: (li * T + t) * 2 + h
    # per-target scalars that do not depend on the response-gene threshold
    tier_e = np.array([ess.get(t, np.nan) for t in targets])
    n_cells = np.zeros((2, T)); n_half = np.zeros((2, T, 2)); resid = np.full((2, T), np.nan); resid_h = np.full((2, T, 2), np.nan)
    for li in range(2):
        for t in range(T):
            c0, c1 = cnt[row(li, t, 0)], cnt[row(li, t, 1)]
            n_cells[li, t] = c0[0] + c1[0]; n_half[li, t] = (c0[0], c1[0])
            j = gpos.get(targets[t])
            if j is not None:
                for h, c in ((0, c0), (1, c1)):
                    e = rate[li][j] * c[4]
                    if e >= 1:
                        resid_h[li, t, h] = c[5] / e
                e = rate[li][j] * (c0[4] + c1[4])
                if e >= 1:
                    resid[li, t] = (c0[5] + c1[5]) / e
    usable = (n_cells >= MIN_CELLS) & (n_half[..., 0] >= 5) & (n_half[..., 1] >= 5)
    usable[:, ct] = False
    inert = np.array([[(gpos.get(t) is not None and cp[li][gpos[t]] < 0.01) for t in targets] for li in range(2)])
    expressed = np.array([[(gpos.get(t) is not None and cp[li][gpos[t]] >= 0.5) for t in targets] for li in range(2)])
    res = {"rules": __doc__, "lines": lines, "n_targets_usable_both": int((usable[0] & usable[1]).sum()),
           "umi_per_cell": {L: float(clib[li] / (cnt[row(li, ct, 0)][0] + cnt[row(li, ct, 1)][0])) for li, L in enumerate(lines)},
           "residual_median_by_tier": {}, "premise_residual_half_corr": {}}
    for li, L in enumerate(lines):
        ok = usable[li] & np.isfinite(resid_h[li, :, 0]) & np.isfinite(resid_h[li, :, 1])
        res["premise_residual_half_corr"][L] = dict(n=int(ok.sum()), spearman=float(_spearman(resid_h[li, ok, 0], resid_h[li, ok, 1])))
        res["residual_median_by_tier"][L] = {nm: float(np.nanmedian(resid[li, usable[li] & np.isfinite(tier_e) & (tier_e >= lo) & (tier_e < hi)]))
                                             for nm, lo, hi in TIERS}
    print("premise", json.dumps(res["premise_residual_half_corr"]), flush=True)
    print("residual medians", json.dumps(res["residual_median_by_tier"]), flush=True)
    rng = np.random.default_rng(1)
    res["sweep"] = {}
    for min_resp in (0.5, 1.0):
        sel = np.where((cp[0] >= min_resp) & (cp[1] >= min_resp))[0]
        pos_in_sel = {g: q for q, g in enumerate(sel)}
        mc = [[np.asarray(S[row(li, ct, h)])[sel] / cnt[row(li, ct, h)][0] for h in (0, 1)] for li in range(2)]
        mag = np.full((2, T), np.nan)
        for li in range(2):
            for t in np.where(usable[li])[0]:
                v = []
                for h in (0, 1):
                    x = np.asarray(S[row(li, t, h)])[sel] / cnt[row(li, t, h)][0] - mc[li][h]
                    q = pos_in_sel.get(gpos.get(targets[t], -1))
                    if q is not None:
                        x[q] = 0
                    v.append(x)
                mag[li, t] = v[0] @ v[1]
        both = usable[0] & usable[1]
        thr = [float(np.nanpercentile(mag[li, both & inert[li]], 97.5)) for li in range(2)]
        disc = [mag[li] > thr[li] for li in range(2)]
        block = {"n_response_genes": int(len(sel)), "inert_threshold": {lines[li]: thr[li] for li in range(2)},
                 "n_inert": {lines[li]: int((both & inert[li]).sum()) for li in range(2)}}
        block["all_targets"] = paired_ratio(mag[1, both], mag[0, both], rng)
        for kd in (0.2, 0.3, 0.5):
            m = both & (resid[0] <= kd) & (resid[1] <= kd)
            block["matched_kd_%.1f" % kd] = paired_ratio(mag[1, m], mag[0, m], rng)
            block["matched_kd_%.1f" % kd]["n_matched"] = int(m.sum())
        # discernible fraction among expressed targets, all vs matched (0.3)
        exp_both = both & expressed[0] & expressed[1]
        m = exp_both & (resid[0] <= 0.3) & (resid[1] <= 0.3)
        block["discernible"] = {L: dict(all=frac_ci(disc[li][exp_both], rng), matched_kd_0_3=frac_ci(disc[li][m], rng),
                                        all_onesided_expressed=frac_ci(disc[li][usable[li] & expressed[li]], rng)) for li, L in enumerate(lines)}
        # per tier, all vs matched 0.3
        block["tiers"] = {}
        for nm, lo, hi in TIERS:
            tm = both & np.isfinite(tier_e) & (tier_e >= lo) & (tier_e < hi)
            mm = tm & (resid[0] <= 0.3) & (resid[1] <= 0.3)
            block["tiers"][nm] = dict(all=paired_ratio(mag[1, tm], mag[0, tm], rng), matched=paired_ratio(mag[1, mm], mag[0, mm], rng),
                                      median_mag={lines[li]: float(np.nanmedian(mag[li, tm])) for li in range(2)},
                                      median_resid={lines[li]: float(np.nanmedian(resid[li, tm])) for li in range(2)},
                                      rel_cells={lines[li]: float(np.median(n_cells[li, tm]) / np.median(n_cells[li, both & (tier_e > -0.2)])) for li in range(2)})
        res["sweep"]["min_resp_%.1f" % min_resp] = block
        print("== min_resp", min_resp, "genes", len(sel), flush=True)
        for k in ("all_targets", "matched_kd_0.2", "matched_kd_0.3", "matched_kd_0.5"):
            print("  ", k, json.dumps(block[k]), flush=True)
        print("   discernible", json.dumps(block["discernible"]), flush=True)
        for nm, v in block["tiers"].items():
            print("   tier", nm, json.dumps(v), flush=True)
        if min_resp == 0.5:
            np.savez_compressed(out + "/per_target.npz", targets=np.array(targets), essentiality=tier_e, mag=mag, resid=resid,
                                resid_half=resid_h, n_cells=n_cells, usable=usable, inert=inert, expressed=expressed)
    json.dump(res, open(out + "/metrics_orion_matched.json", "w"), indent=1)


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    main()
