"""Export everything the fate paper's figures need from the A100 outputs into one JSON (2026-10-18).
Figures and panels follow notes/fate_paper_figures.md §4; every number must already exist in
notes/fate_paper_ledger.md, so this script only reshapes saved outputs - it computes no new estimates except the
per-target arrays the panels plot.
Usage: fate_figdata_a100.py <out json>."""
import json, sys
import numpy as np
import config

F = config.get("FATE_ROOT")
TIERS = ("lt_-1", "-1_-0.5", "-0.5_-0.2", "gt_-0.2")


def main():
    out = {"rules": __doc__}
    out["ci"] = json.load(open(F + "/ci/fate_ci.json"))

    # Fig 1d: hESC library normalisation
    L = json.load(open(F + "/pan_libnorm/pan_library_norm.json"))
    out["fig1d"] = dict(tiers=L["tiers"], copies_bins=L["copies_bins"],
                        rows=[dict(e=r["essentiality"], raw=r["raw"], norm=r["norm"], copies=r["copies_per_guide"],
                                   resid=r["residual"], parity=r["parity_norm"])
                              for r in L["rows"] if np.isfinite(r.get("norm", np.nan))])

    # Fig 1c: dG1 by tier, RPE1 / K562 (budget rows)
    M = json.load(open(F + "/budget/metrics_fate_budget.json"))
    out["fig1c"] = {}
    for lab in ("K562", "RPE1", "hESC"):
        R = M[lab]["rows_random"]
        out["fig1c"][lab] = [dict(e=r["essentiality"], dG1=r["dG1"], comp=r["m_comp"], tot=r["m_total"],
                                  kd=r.get("m_kd"), resid=r["resid"], cells=r["cells"]) for r in R.values()]
    out["fig1c"]["orion"] = json.load(open(F + "/orion_c1/metrics_orion_c1.json"))
    out["fig1c"]["leng"] = {k: v for k, v in json.load(open(F + "/leng/metrics_leng.json")).items() if k != "rules"}

    # Fig 2b, 2c: same-scale per target and the dilution model
    z = np.load(F + "/samescale_out/per_target.npz", allow_pickle=True)
    good = (z["res_h"] <= 0.3) & (z["res_6"] <= 0.3) & (z["res_8"] <= 0.3)
    out["fig2b"] = dict(targets=[str(t) for t in z["targets"]], e=z["essentiality"].tolist(),
                        h=z["h_guide_lfc"].tolist(), k=z["k_cross_lfc"].tolist(), good=good.tolist(),
                        res_h=z["res_h"].tolist(), res_6=z["res_6"].tolist(), res_8=z["res_8"].tolist(),
                        robustness=json.load(open(F + "/samescale_rob/metrics_fate_samescale.json"))
                        if __import__("os").path.exists(F + "/samescale_rob/metrics_fate_samescale.json") else None)
    out["fig2c"] = json.load(open(F + "/dilution_model/metrics_dilution_model.json"))
    out["fig2d"] = json.load(open(F + "/kdgroup/metrics_kdgroup.json"))
    out["fig2d"].pop("rules", None)

    # Fig 3: budget summaries incl. cycle-gene-removed and negative controls
    out["fig3"] = {}
    for path, labs in ((F + "/budget/metrics_fate_budget.json", ("K562", "RPE1", "hESC")),
                       (F + "/budget_hesc_nc/metrics_fate_budget.json", ("hESC",)),
                       (F + "/budget_cd4/nc/metrics_fate_budget.json", ("CD4_D2rest",)),
                       (F + "/budget_cd4/d34/metrics_fate_budget.json", ("CD4_D3rest", "CD4_D4rest"))):
        try:
            m = json.load(open(path))
        except FileNotFoundError:
            continue
        for lab in labs:
            if lab not in m:
                continue
            key = lab if path.endswith("budget/metrics_fate_budget.json") else lab + "|" + path.split("/")[-2]
            out["fig3"][key] = {k: m[lab][k] for k in ("meta", "random", "guide", "random_no_cycle_genes",
                                                       "negative_control", "T3_calibration") if k in m[lab]}
    S = json.load(open(F + "/budget_d8/metrics_budget_screens.json"))
    out["fig3"]["K562_screens"] = dict(tiers=S["tiers"], negative_control=S["negative_control"])

    # Fig 5: cross-context, failing nulls, denoised benchmark
    out["fig5a"] = json.load(open(F + "/xctx/metrics_crosscontext_K562.json"))
    out["fig5a"].pop("rules", None)
    out["fig5a_null"] = json.load(open(F + "/xctx/metrics_c6_null_K562.json"))
    out["fig5a_null"].pop("rules", None)
    out["fig5c"] = {s: json.load(open("%s/xctx/w5_benchmark_denoised_%s.json" % (F, s))) for s in
                    ("K562", "RPE1", "Jurkat", "HepG2")}

    # Fig 6: spread vs error, Jost, programmes
    out["fig6a"] = json.load(open(F + "/jost/metrics_c7_programs.json"))
    out["fig6a"].pop("rules", None)
    J = json.load(open(F + "/jost/metrics_fate_jost.json"))
    out["fig6b_jost"] = dict(summary=J["summary"],
                             genes={g: dict(gamma_top=v["gamma_top"], D=v["D"], D_se=v["D_se"], D_loo=v["D_loo"],
                                            r_top=v["r_top"], r_top_se=v["r_top_se"], n_guides=v["n_guides"],
                                            activity=v["activity"], r=v["r"], p=v["p"])
                                    for g, v in J["genes"].items()})
    out["fig6b_prog"] = json.load(open(F + "/jost/metrics_c7_posthoc.json"))
    out["fig6b_prog"].pop("rules", None)

    json.dump(out, open(sys.argv[1], "w"))
    print("written", sys.argv[1], "top-level keys", len(out), flush=True)


if __name__ == "__main__":
    main()
