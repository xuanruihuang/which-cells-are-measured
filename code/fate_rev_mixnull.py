"""Fate v3r2 external review, check V4, analysis (expectation pre-stated in notes/fate_analysis_redesign.md §7.2, commit
f9d9a81). A100. Inputs: pan_extract_mixnull.py output, pan_agg_kdgroup (knocked-down sums, controls), fate_kdgroup.py
rows (m_kd_corr with the e = 0 thinning null), the mixture rows of fate_mixture.py (e per target). Genes as
fate_samescale.py. Per analysable target: magnitude of the mixed null at e = 0, 0.3, 0.5 (cross-signal between guide
parities, each against the controls of its parity); knocked-down magnitude corrected with each null; the null at the e
nearest the target's fitted e. Reported relative to the knocked-down signal, per stratum.
Usage: fate_rev_mixnull.py <out json>. Exploratory."""
import sys, json
import numpy as np
import config
from fate_samescale import load_k562

FI = config.get("FATE_ROOT"); KDG = config.get("PAN_KDGROUP"); PGR = config.get("PAN_GUIDE_RAW")
MIN_CELLS = 10


def lfc(rt, lt, rc, lc):
    return np.log2((rt + 0.5) / lt) - np.log2((rc + 0.5) / lc)


def bm(x, rng):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return None
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(2000)]
    return [float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(len(x))]


def main():
    out = sys.argv[1]; rng = np.random.default_rng(3)
    idx = json.load(open(KDG + "/index.json")); pt, pg = idx["targets"], idx["genes"]; T, G = len(pt), len(pg)
    R = np.load(KDG + "/raw_sums.npy", mmap_mode="r").reshape(T, 2, 3, G); C = np.load(KDG + "/counts.npy").reshape(T, 2, 3, 4)
    craw, clib = np.load(KDG + "/ctrl_raw.npy"), np.load(KDG + "/ctrl_lib.npy")
    mi = json.load(open(FI + "/mixnull/index.json")); EM = mi["emix"]; nidx = {t: k for k, t in enumerate(mi["null_targets"])}
    MR = np.load(FI + "/mixnull/mixnull_raw.npy", mmap_mode="r"); MC = np.load(FI + "/mixnull/mixnull_cnt.npy")
    praw = np.load(PGR + "/raw_sums.npy", mmap_mode="r").reshape(T, 4, G); pcnt = np.load(PGR + "/counts.npy").reshape(T, 4, 6)
    K6, K8 = load_k562(FI + "/samescale2/K6.npz"), load_k562(FI + "/samescale2/K8.npz")
    gset = sorted(set(K6["genes"]) & set(K8["genes"])); pgi = {g: j for j, g in enumerate(pg)}
    k6i, k8i = {g: j for j, g in enumerate(K6["genes"])}, {g: j for j, g in enumerate(K8["genes"])}
    ct = pt.index("non-targeting")
    hc = np.array([praw[ct, :, pgi[g]].sum() for g in gset]) / pcnt[ct, :, 4].sum() * 1e4
    c6 = np.array([K6["raw"][-1, :, k6i[g]].sum() for g in gset]) / K6["lib"][-1].sum() * 1e4
    c8 = np.array([K8["raw"][-1, :, k8i[g]].sum() for g in gset]) / K8["lib"][-1].sum() * 1e4
    genes = [gset[j] for j in np.where((hc >= 1) & (c6 >= 1) & (c8 >= 1))[0]]; gq = {g: q for q, g in enumerate(genes)}
    hcol = np.array([pgi[g] for g in genes])
    rows = json.load(open(FI + "/kdgroup/metrics_kdgroup.json"))["rows"]
    mix = {r["target"]: r for r in json.load(open(FI + "/mixture/metrics_mixture.json"))["rows"]["hESC"]}

    def zero(v, t):
        q = gq.get(t)
        if q is not None:
            v = v.copy(); v[q] = 0
        return v
    out_rows = []
    for r in rows:
        t = r["target"]
        if t not in nidx or not np.isfinite(r["m_kd_corr"]):
            continue
        k = nidx[t]; ti = pt.index(t)
        if C[ti, 0, 0, 0] < MIN_CELLS or C[ti, 1, 0, 0] < MIN_CELLS:
            continue
        vk = [zero(lfc(R[ti, p, 0][hcol], C[ti, p, 0, 1], craw[p, hcol], clib[p]), t) for p in (0, 1)]
        d = dict(target=t, essentiality=r["essentiality"], m_kd_corr=r["m_kd_corr"], e_fit=mix[t]["nb"]["e"] if t in mix else None)
        for i, e in enumerate(EM):
            if MC[i, 0, k, 0] < MIN_CELLS or MC[i, 1, k, 0] < MIN_CELLS:
                continue
            nv = [zero(lfc(MR[i, p, k][hcol], MC[i, p, k, 1], craw[p, hcol], clib[p]), t) for p in (0, 1)]
            d["null_%g" % e] = float(nv[0] @ nv[1]); d["m_kd_corr_%g" % e] = float((vk[0] - nv[0]) @ (vk[1] - nv[1]))
            d["cells_%g" % e] = float(MC[i, :, k, 0].sum())
        if d["e_fit"] is not None:
            e_near = min(EM, key=lambda e: abs(e - d["e_fit"]))
            d["m_kd_corr_near"] = d.get("m_kd_corr_%g" % e_near)
        out_rows.append(d)
    res = {"rules": __doc__, "rows": out_rows, "tiers": {}}
    for nm, lo, hi in (("lt_-1", -9, -1), ("-1_-0.5", -1, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, 9), ("all", -99, 99)):
        rr = [d for d in out_rows if d["essentiality"] is not None and np.isfinite(d["essentiality"]) and lo <= d["essentiality"] < hi]
        s = dict(n=len(rr), m_kd_corr=bm([d["m_kd_corr"] for d in rr], rng))
        for e in EM:
            s["null_%g_over_signal" % e] = bm([d.get("null_%g" % e, np.nan) / d["m_kd_corr"] for d in rr if d["m_kd_corr"] > 0], rng)
            s["corrected_%g_over_null2_corrected" % e] = bm([d.get("m_kd_corr_%g" % e, np.nan) / d["m_kd_corr"] for d in rr if d["m_kd_corr"] > 0], rng)
        s["near_e_over_null2_corrected"] = bm([d.get("m_kd_corr_near", np.nan) / d["m_kd_corr"] for d in rr if d["m_kd_corr"] > 0], rng)
        res["tiers"][nm] = s
        print(nm, json.dumps(s), flush=True)
    json.dump(res, open(out, "w"), indent=1, default=lambda v: None if v is None else float(v))


if __name__ == "__main__":
    main()
