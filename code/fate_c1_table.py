"""Unified fate table across contexts (5090; fate paper C1, 2026-10-18). One essentiality scale for every context: median
DepMap 24Q4 gene effect over all lines, four tiers (< -1, -1 to -0.5, -0.5 to -0.2, > -0.2). Per context and tier, from
per-target tables already computed by each context's script:
  rel_cells   median cells per target / median of the non-essential tier (raw; library composition not removed unless
              stated: hESC also library-input normalized, pan_library_norm.py)
  residual    median knockdown residual of the survivors (own-target raw / expected), targets as in each source table
  dG1         median G1-fraction change vs controls
  response    median response per tier / median of the non-essential tier; unit = "independent" (guide or construct
              halves, or independent screens) or "random" (random cell halves; inflated by clonal energy, compresses the
              ratio toward 1 in contexts with strong clonal structure, findings ah, ba)
Sources (per-target): explore_W pert tables (K562 day-6 essential, RPE1, Jurkat, HepG2, CD4 donor 1 rest / stim 48 h),
pan_fate (hESC), feng_fate pooled (iPSC day 3), fate_h1 (VCC H1, three splits pooled), fate_maffa (16 lines pooled).
Tier-level independent-unit responses: fate_iu_cd4 (guide halves, donors 1-2), fate_iu_pan (guide halves).
Usage: fate_c1_table.py <depmap csv> <inputs dir> <out_dir>. Exploratory."""
import os, sys, json
import numpy as np
import config

TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf))
BASE = config.get("CC_ROOT")


def ess_median(path):
    with open(path) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        rows = [np.array([float(x) if x else np.nan for x in line.rstrip("\n").split(",")[1:]]) for line in f]
    return dict(zip(sym, np.nanmedian(np.array(rows), 0)))


def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def summarize(names, cols, ess, unit):
    e = np.array([ess.get(str(t), np.nan) for t in names])
    out = {}
    for nm, lo, hi in TIERS:
        m = np.isfinite(e) & (e >= lo) & (e < hi)
        out[nm] = {"n": int(m.sum())}
        for k, v in cols.items():
            out[nm][k] = med(np.asarray(v, float)[m]) if v is not None else None
    ne = out["gt_-0.2"]
    for nm in out:
        for k, kk in (("cells", "rel_cells"), ("response", "response_ratio"), ("norm", "rel_norm")):
            if out[nm].get(k) is not None and ne.get(k):
                out[nm][kk] = out[nm][k] / ne[k]
    out["response_unit"] = unit
    return out


def main():
    dep, inp, outd = sys.argv[1:4]; os.makedirs(outd, exist_ok=True)
    ess = ess_median(dep)
    T = {}
    for lab, f in (("K562 (Replogle essential)", "pert_table_k562"), ("RPE1 (Replogle essential)", "pert_table_rpe1"),
                   ("Jurkat (Nadig)", "pert_table_jurkat"), ("HepG2 (Nadig)", "pert_table_hepg2")):
        z = np.load("%s/explore_W/%s.npz" % (BASE, f), allow_pickle=True)
        T[lab] = summarize(z["pert"], dict(cells=z["n"], residual=z["kd_resid"], dG1=z["dG1"], response=z["energy"]), ess, "random")
    for lab, f in (("CD4 T rest (Zhu, donor 1)", "cd4_rest_pert_table"), ("CD4 T stim 48 h (Zhu, donor 1)", "cd4_stim48_pert_table")):
        z = np.load("%s/explore_W/%s.npz" % (BASE, f), allow_pickle=True)
        T[lab] = summarize(z["group"], dict(cells=z["n"], residual=z["kd_resid"], dG1=z["dG1"], response=z["energy"]), ess, "random")
    z = np.load(inp + "/pan_fate_pert_table_hesc.npz", allow_pickle=True)
    ln = json.load(open(inp + "/fate_iu_pan_libnorm_pan_library_norm.json"))["rows"]
    nmap = {r["target"]: r["norm"] for r in ln}
    T["hESC (Pan)"] = summarize(z["target"], dict(cells=z["n"], residual=z["kd_resid"], dG1=z["dG1"], response=z["energy"],
                                                 norm=[nmap.get(str(t), np.nan) for t in z["target"]]), ess, "random")
    z = np.load(inp + "/feng_fate_pert_table_ipsc_pooled.npz", allow_pickle=True)
    T["iPSC day 3 (Feng, 12 lines pooled)"] = summarize(z["target"], dict(cells=z["n"], residual=z["kd_resid"], dG1=z["dG1"], response=z["energy"]), ess, "random")
    h1 = json.load(open(BASE + "/fate_h1/metrics_fate_h1.json"))
    rows = [r for s in ("Training", "Validation", "Test") for r in h1[s]["rows"]]
    T["H1 hESC (VCC 2025)"] = summarize([r["target"] for r in rows], dict(cells=[r["cells"] for r in rows], residual=[r["residual"] for r in rows],
                                        dG1=[r["dG1"] for r in rows], response=[r["response"] for r in rows]), ess, "random")
    mf = json.load(open(BASE + "/fate_maffa/metrics_fate_maffa.json"))["rows"]
    T["16 cancer lines KO day 6 (Maffa, pooled)"] = summarize([r["gene"] for r in mf], dict(cells=[r["depletion"] for r in mf], residual=None, dG1=None,
                                                              response=[r["response"] for r in mf]), ess, "independent (construct halves)")
    # independent-unit responses at tier level
    iu = {}
    cd = json.load(open(inp + "/fate_iu_cd4_metrics_fate_iu_cd4.json"))
    for k in ("D1_rest", "D1_stim", "D2_rest", "D2_stim"):
        tt = cd[k]["tiers"]
        iu["CD4 " + k] = {nm: tt[nm]["resp_guide_median"] / tt["gt_-0.2"]["resp_guide_median"] for nm in tt if "resp_guide_median" in tt[nm]}
    pn = json.load(open(inp + "/fate_iu_pan_metrics_fate_iu_pan.json"))["tiers"]
    iu["hESC"] = {nm: pn[nm]["resp_guide_median"] / pn["gt_-0.2"]["resp_guide_median"] for nm, *_ in TIERS}
    # library-free cross-context depletion: K562 essential, RPE1, Jurkat, HepG2 share the growth-balanced essential
    # library (Replogle 2022; Nadig 2025: dJR092), so per-target share ratios between contexts cancel the library weights.
    shares = {}
    for lab, f in (("K562", "pert_table_k562"), ("RPE1", "pert_table_rpe1"), ("Jurkat", "pert_table_jurkat"), ("HepG2", "pert_table_hepg2")):
        z = np.load("%s/explore_W/%s.npz" % (BASE, f), allow_pickle=True)
        names = [str(p) for p in z["pert"]]; n = z["n"].astype(float)
        e = np.array([ess.get(p, np.nan) for p in names]); ne = np.isfinite(e) & (e > -0.2)
        shares[lab] = dict(zip(names, n / n[ne].sum()))
    cross = {}
    for lab in ("RPE1", "Jurkat", "HepG2"):
        common = sorted(set(shares[lab]) & set(shares["K562"]))
        r = np.array([shares[lab][t] / shares["K562"][t] for t in common]); e = np.array([ess.get(t, np.nan) for t in common])
        r = r / med(r[np.isfinite(e) & (e > -0.2)])      # relative to the shared non-essential targets
        cross[lab + "_vs_K562"] = {nm: dict(n=int(((e >= lo) & (e < hi)).sum()), median_share_ratio=med(r[(e >= lo) & (e < hi)]))
                                   for nm, lo, hi in TIERS}
    res = {"rules": __doc__, "contexts": T, "independent_unit_response_ratio": iu, "cross_context_share_vs_K562": cross}
    for k, v in cross.items():
        print("CROSS", k, {a: (b["n"], round(b["median_share_ratio"], 2) if b["median_share_ratio"] else None) for a, b in v.items()})
    json.dump(res, open(outd + "/c1_table.json", "w"), indent=1)
    f = lambda v, p="%.2f": (p % v) if isinstance(v, (int, float)) and v is not None and np.isfinite(v) else "-"
    for lab, t in T.items():
        print("==", lab, "| response unit:", t["response_unit"])
        for nm, *_ in TIERS:
            r = t[nm]
            print("   %-10s n=%4d rel_cells=%s rel_norm=%s residual=%s dG1=%s resp_ratio=%s" % (
                nm, r["n"], f(r.get("rel_cells")), f(r.get("rel_norm"), "%.3f"), f(r.get("residual")), f(r.get("dG1"), "%.3f"), f(r.get("response_ratio"))))
    for k, v in iu.items():
        print("IU", k, {a: round(b, 2) for a, b in v.items()})


if __name__ == "__main__":
    main()
