"""Library-input normalization of hESC cell depletion (A100; fate paper C1, 2026-10-18). Pan et al. 2026 over-represented
sgRNAs in the oligo pool in proportion to their expected dropout in a balancing screen (Methods: an sgRNA expected to
drop out 4-fold was 4-fold over-represented). Supplementary Table 6 lists one row per oligo copy (<element>_f_<k>_left);
copies per element were tabulated locally into <copies tsv> (element, copies; 1-32, non-targeting 1). Raw cell numbers
therefore understate elimination.
Join: element names = CRISPR Guide Capture feature names of the 10x files; guide parity as in pan_extract_guide.py (rank
of the guide name among the target's guides, mod 2). Cells, library and own-target raw UMIs per (target, parity) from
pan_extract_guide.py raw output (counts.npy, key = target * 4 + parity * 2 + half).
  normalized abundance a = (cells / copies) / (non-targeting cells / non-targeting copies)
  raw abundance        r = (cells / guides) / (non-targeting cells / non-targeting guides)
Assumes plasmid representation proportional to oligo copies (Twist synthesis, > 1000 colonies per element).
Knockdown residual = own-target raw / (control rate x library), control rate pooled over lanes (as fate_iu_pan.py).
Per DepMap tier: medians of r, a, residual; Spearman of copies with essentiality; r by copies bin; residual vs a.
Usage: pan_library_norm.py <raw_dir> <pan_agg_guide_raw dir> <copies tsv> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr
from pan_extract import read_lane, setup
from strata import essentiality, TIERS

BINS = ((1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 1000))


def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def main():
    raw, d, ctsv, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
    copies = {l.split("\t")[0]: int(l.split("\t")[1]) for l in open(ctsv) if l.strip()}
    S = setup(raw)
    ft, names, *_ = read_lane(S["files"][0])
    guides = np.array([str(g) for g in names[S["guide_rows"]]])
    gti = S["guide_target_idx"]; T = len(S["targets"])
    par = np.zeros(len(guides), np.int64)
    for t in range(T):
        ix = np.where(gti == t)[0]
        for r, k in enumerate(ix[np.argsort(guides[ix])]):
            par[k] = r % 2
    idx = json.load(open(d + "/index.json")); assert idx["targets"] == S["targets"]
    cnt = np.load(d + "/counts.npy").reshape(T, 2, 2, 6).sum(2)          # target x parity x [n, G1, S, G2M, lib, own]
    gc = np.array([copies.get(g, -1) for g in guides]); miss = guides[gc < 0]
    print("guides", len(guides), "matched", int((gc > 0).sum()), "unmatched", len(miss), list(miss[:5]), flush=True)
    C = np.zeros((T, 2)); NG = np.zeros((T, 2)); bad = np.zeros((T, 2), bool)
    for k in range(len(guides)):
        if gc[k] < 0:
            bad[gti[k], par[k]] = True
        else:
            C[gti[k], par[k]] += gc[k]
        NG[gti[k], par[k]] += 1
    ct = S["ctrl_t"]; n = cnt[:, :, 0]
    N0, C0, G0 = n[ct].sum(), C[ct].sum(), NG[ct].sum()
    genes = idx["genes"]; gpos = {g: j for j, g in enumerate(genes)}
    lam = np.load(d + "/ctrl_raw.npy").sum(0) / np.load(d + "/ctrl_lib.npy").sum()
    ess = essentiality()
    rows = []
    for t, name in enumerate(S["targets"]):
        if t == ct or bad[t].any() or NG[t].sum() == 0:
            continue
        nt, ctot, gtot = n[t].sum(), C[t].sum(), NG[t].sum()
        j = gpos.get(name)
        res = float(cnt[t, :, 5].sum() / (lam[j] * cnt[t, :, 4].sum())) if j is not None and lam[j] > 0 and cnt[t, :, 4].sum() > 0 else np.nan
        expd = bool(j is not None and lam[j] * 1e4 >= 0.5)                 # control >= 0.5 CP10K
        rows.append(dict(target=name, cells=int(nt), copies=int(ctot), guides=int(gtot),
                         copies_per_guide=float(ctot / gtot), raw=float((nt / gtot) / (N0 / G0)),
                         norm=float((nt / ctot) / (N0 / C0)) if ctot > 0 else np.nan,
                         parity_norm=[float((n[t, p] / C[t, p]) / (N0 / C0)) if C[t, p] > 0 else None for p in (0, 1)],
                         residual=res, expressed=expd, essentiality=float(ess.get(name, np.nan))))
    e = np.array([r["essentiality"] for r in rows]); cpg = np.array([r["copies_per_guide"] for r in rows])
    rr = np.array([r["raw"] for r in rows]); aa = np.array([r["norm"] for r in rows])
    rs = np.array([r["residual"] for r in rows]); ex = np.array([r["expressed"] for r in rows])
    fin = np.isfinite(e)
    summ = dict(n_targets=len(rows), nt_cells=int(N0), nt_copies=int(C0), nt_guides=int(G0),
                spearman_copies_essentiality=float(spearmanr(cpg[fin], e[fin])[0]),
                spearman_raw_essentiality=float(spearmanr(rr[fin], e[fin])[0]),
                spearman_norm_essentiality=float(spearmanr(aa[fin], e[fin])[0]))
    tiers = {}
    for nm, lo, hi in TIERS:
        m = fin & (e >= lo) & (e < hi)
        me = m & ex & np.isfinite(rs)
        tiers[nm] = dict(n=int(m.sum()), copies_per_guide_median=med(cpg[m]), raw_median=med(rr[m]), norm_median=med(aa[m]),
                         norm_q25=float(np.quantile(aa[m], 0.25)) if m.any() else None,
                         frac_norm_below_0p1=float((aa[m] < 0.1).mean()) if m.any() else None,
                         residual_median_expressed=med(rs[me]),
                         spearman_residual_norm_expressed=float(spearmanr(rs[me], aa[me])[0]) if me.sum() > 10 else None)
    bins = {}
    for lo, hi in BINS:
        m = (cpg >= lo) & (cpg <= hi)
        bins["%d-%d" % (lo, hi)] = dict(n=int(m.sum()), raw_median=med(rr[m]), norm_median=med(aa[m]),
                                        essentiality_median=med(e[m & fin]))
    me = ex & np.isfinite(rs)
    summ["spearman_residual_norm_expressed_all"] = float(spearmanr(rs[me], aa[me])[0])
    res = dict(rules=__doc__, summary=summ, tiers=tiers, copies_bins=bins, unmatched=list(miss), rows=rows)
    json.dump(res, open(out + "/pan_library_norm.json", "w"), indent=1)
    print(json.dumps(summ), flush=True)
    for k, v in tiers.items():
        print("tier", k, json.dumps(v), flush=True)
    for k, v in bins.items():
        print("copies", k, json.dumps(v), flush=True)


if __name__ == "__main__":
    main()
