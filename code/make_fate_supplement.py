"""Supplementary tables S1-S4 (CSV + a markdown index) and figures S1-S3 of the fate-layer paper (2026-10-18).
Inputs are the exported analysis outputs only (code/a100/fate_figdata_a100.py, code/fate_figdata_5090.py,
code/a100/fate_suppdata.py); nothing is recomputed here except simple reshaping, so every value traces to
notes/fate_paper_ledger.md. S5 (pre-stated expectations) and S6 (number ledger) are curated markdown written
alongside this script.
Usage: make_fate_supplement.py <figdata dir> <out dir>.
"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, BLUE, GREY = "#c0392b", "#2c3e50", "#7f8c8d"
ARREST, TOL, ELIM, ORANGE = "#2980b9", "#95a5a6", "#8e44ad", "#e67e22"
MM = 1 / 25.4
TIERS = ["lt_-1", "-1_-0.5", "-0.5_-0.2", "gt_-0.2"]
TIERLAB = ["< -1", "-1 to -0.5", "-0.5 to -0.2", "> -0.2"]

plt.rcParams.update({"font.size": 7, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "axes.linewidth": 0.6, "axes.labelsize": 7.5, "axes.titlesize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42})


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel(ax, letter, dx=-0.16, dy=1.05):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def write_csv(out, name, header, rows):
    path = os.path.join(out, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else (("%.6g" % v) if isinstance(v, float) else v) for v in r])
    print("wrote", name, len(rows), "rows", flush=True)


def fmt_ci(d):
    if not d or d.get("ci") is None:
        return None, None, None
    return d["est"], d["ci"][0], d["ci"][1]


# ------------------------------------------------------------------ S1
def table_s1(A, B, out):
    """Fate signatures: every context x stratum, with n and harvest day."""
    header = ["context", "fate", "harvest_day", "replication_unit", "stratum", "n_targets", "relative_cells",
              "survivor_residual", "dG1", "response_ratio_vs_nonessential", "note"]
    rows = []
    c1 = B["c1_table"]["contexts"]
    day = {"K562 (Replogle essential)": "6", "RPE1 (Replogle essential)": "7", "Jurkat (Nadig)": "7",
           "HepG2 (Nadig)": "7", "CD4 T rest (Zhu, donor 1)": "~11", "CD4 T stim 48 h (Zhu, donor 1)": "~13",
           "hESC (Pan)": "8", "iPSC day 3 (Feng, 12 lines pooled)": "3", "H1 hESC (VCC 2025)": "unpublished",
           "16 cancer lines KO day 6 (Maffa, pooled)": "6"}
    fate = {"K562 (Replogle essential)": "tolerance", "RPE1 (Replogle essential)": "arrest",
            "Jurkat (Nadig)": "tolerance", "HepG2 (Nadig)": "arrest", "CD4 T rest (Zhu, donor 1)": "elimination",
            "CD4 T stim 48 h (Zhu, donor 1)": "elimination", "hESC (Pan)": "elimination",
            "iPSC day 3 (Feng, 12 lines pooled)": "not yet depleting", "H1 hESC (VCC 2025)": "atypical",
            "16 cancer lines KO day 6 (Maffa, pooled)": "depleting, not diluted"}
    iu = B["c1_table"].get("independent_unit_response_ratio", {})
    for name, d in c1.items():
        tiers = d.get("tiers", d)
        for t, lab in zip(TIERS, TIERLAB):
            v = tiers.get(t, {})
            unit = ("independent constructs" if "Maffa" in name else
                    ("guide halves" if ("hESC (Pan)" in name or "CD4" in name) else "random halves"))
            ratio = v.get("response_ratio", v.get("resp_ratio"))
            if unit == "guide halves":
                key = next((k for k in iu if k.split()[0].lower() in name.lower()
                            and (("rest" in k.lower()) == ("rest" in name.lower()))), None)
                if key and t in iu[key]:
                    ratio = iu[key][t]
            rows.append([name, fate.get(name, ""), day.get(name, ""), unit,
                         lab, v.get("n"), v.get("rel_cells"), v.get("residual", v.get("resid")), v.get("dG1"), ratio,
                         "library balanced against dropout; depletion is a lower bound"
                         if name.startswith(("K562", "RPE1", "Jurkat", "HepG2")) else ""])
    for line, d in A["fig1c"]["orion"].items():
        if not isinstance(d, dict) or "tiers" not in d:
            continue
        for t, lab in zip(TIERS, TIERLAB):
            v = d["tiers"][t]
            rows.append(["%s (X-Atlas/Orion)" % line, "weak arrest" if line == "HCT116" else "tolerance",
                         "7", "random halves", lab, v.get("n"), v.get("rel_cells"), v.get("resid"),
                         v.get("dG1"), v.get("resp_ratio"), "control phase fractions in Figure S1"])
    for state, d in A["fig1c"]["leng"]["states"].items():
        for tname, v in d["targets"].items():
            if not np.isfinite(v.get("depmap", np.nan)) or v["depmap"] > -0.5:
                continue
            rows.append(["iPSC-derived astrocytes (Leng), %s" % state, "no depletion", "13", "none available",
                         "target %s (DepMap %.2f)" % (tname, v["depmap"]), v["cells"], v["rel_cells"],
                         v.get("resid"), None, None, "low-dividing context; descriptive only"])
    write_csv(out, "TableS1_fate_signatures.csv", header, rows)
    return len(rows)


# ------------------------------------------------------------------ S2
def table_s2(A, S, out):
    """Same-scale comparison: per-target values and the threshold sweep."""
    d = A["fig2b"]
    header = ["target", "depmap_gene_effect", "hESC_magnitude_guide_halves", "K562_magnitude_cross_screen",
              "log2_ratio", "residual_hESC", "residual_K562_day6", "residual_K562_day8", "both_sides_good_knockdown"]
    rows = []
    for i, t in enumerate(d["targets"]):
        h, k = d["h"][i], d["k"][i]
        lr = float(np.log2(h / k)) if (np.isfinite(h) and np.isfinite(k) and h > 0 and k > 0) else None
        rows.append([t, d["e"][i], h, k, lr, d["res_h"][i], d["res_6"][i], d["res_8"][i],
                     "yes" if d["good"][i] else "no"])
    write_csv(out, "TableS2_samescale_per_target.csv", header, rows)
    sweep = [["knockdown threshold (both sides)", r["knockdown_threshold"], "matched targets", r["n_matched"],
              "plotted (positive both sides)", r["n_plotted"], "ratio", r["ratio"]]
             for r in S["threshold_sweep_knockdown_only"]]
    write_csv(out, "TableS2b_threshold_sweep.csv",
              ["quantity", "value", "quantity", "value", "quantity", "value", "quantity", "value"], sweep)
    return len(rows)


# ------------------------------------------------------------------ S3
def table_s3(A, B, out):
    """Budget: every context and stratum, with intervals, cycle-gene-removed values and the null control."""
    header = ["context", "design", "stratum", "n_targets", "composition_share", "comp_ci_lo", "comp_ci_hi",
              "within_share", "selection_term", "selection_ci_lo", "selection_ci_hi", "n_classifiable",
              "composition_share_cycle_genes_removed", "median_total_magnitude", "survivor_residual",
              "inert_control_ratio", "used_in_figure3"]
    ci = A["ci"]["bm"]; ci5 = B["ci_5090"]["bm"]; civ = A["ci"]["bv"]
    src = {"K562": A["fig3"]["K562"], "RPE1": A["fig3"]["RPE1"], "hESC": A["fig3"]["hESC"],
           "Jurkat": B["budget"]["Jurkat"], "HepG2": B["budget"]["HepG2"],
           "CD4_D2rest": A["fig3"]["CD4_D2rest|nc"], "CD4_D3rest": A["fig3"]["CD4_D3rest|d34"],
           "CD4_D4rest": A["fig3"]["CD4_D4rest|d34"]}
    cis = dict(ci); cis.update({k: ci5[k] for k in ("Jurkat", "HepG2") if k in ci5})
    rows = []
    for lab, m in src.items():
        nc = m.get("negative_control", {})
        for design in ("random", "guide"):
            if design not in m:
                continue
            for t, tl in zip(TIERS, TIERLAB):
                v = m[design].get(t, {})
                c = (cis.get(lab, {}).get(t, {}) if design == "random" else {})
                ce, clo, chi = fmt_ci(c.get("comp_share"))
                se, slo, shi = fmt_ci(c.get("selection"))
                used = "yes" if (design == "random" and lab in ("K562", "RPE1", "hESC", "Jurkat", "HepG2")) else "no"
                rows.append([lab, design + " halves", tl, v.get("n"), v.get("comp_share"), clo, chi,
                             v.get("within_share"), v.get("selection_kd_over_total"), slo, shi, v.get("n_kd"),
                             m.get("random_no_cycle_genes", {}).get(t, {}).get("comp_share") if design == "random" else None,
                             v.get("m_total_median"), v.get("resid"),
                             nc.get("ratio") if design == "random" else nc.get("guide_ratio"), used])
    K = A["fig3"]["K562_screens"]
    for t, tl in zip(TIERS, TIERLAB):
        v = K["tiers"][t]; ce, clo, chi = fmt_ci(civ[t]["comp_share"]); se, slo, shi = fmt_ci(civ[t]["selection"])
        rows.append(["K562", "two independent screens", tl, v.get("n"), v.get("comp_share"), clo, chi,
                     v.get("within_share"), v.get("selection_kd_over_total"), slo, shi, v.get("n_kd"), None,
                     v.get("m_total_median"), v.get("resid_d6"), None, "yes"])
    write_csv(out, "TableS3_budget.csv", header, rows)
    return len(rows)


# ------------------------------------------------------------------ S4
def table_s4(A, out):
    """Benchmark metrics by source, ground-truth definition and stratum, raw and denoised."""
    header = ["source_context", "target_context", "ground_truth", "subset", "stratum", "n_targets",
              "noise_share_of_truth", "scale_factor_raw", "relative_mse_raw", "scale_factor_denoised",
              "relative_mse_denoised", "deattenuated_cosine"]
    rows = []
    for src, d in A["fig5c"].items():
        for tgt, v in d.items():
            for key, val in v.items():
                kind, subset = key.rsplit("_", 1)
                for t, tl in zip(TIERS, TIERLAB):
                    r = val.get(t, {})
                    if "naive_scale" not in r:
                        continue
                    rows.append([src, tgt, "total effect" if kind == "total" else "silenced cells only",
                                 "all available" if subset == "all" else "common subset", tl, r.get("n"),
                                 r.get("noise_share_median"), r.get("naive_scale"), r.get("naive_rel_mse"),
                                 r.get("denoised_scale"), r.get("denoised_rel_mse"), r.get("deatt_cosine")])
    write_csv(out, "TableS4_benchmark.csv", header, rows)
    return len(rows)


# ------------------------------------------------------------------ Figure S1
def figure_s1(A, B, S, S5, out):
    fig = plt.figure(figsize=(183 * MM, 85 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.08, right=0.98, top=0.86, bottom=0.26)
    ph = dict(S["phase"])
    ph.update(S5["phase"])
    for lab, d in A["fig1c"]["orion"].items():
        if isinstance(d, dict) and "control_phase" in d:
            ph[lab] = dict(d["control_phase"], control_cells=None, proliferation_cp10k={})
    leng = A["fig1c"]["leng"]["states"]
    for st, d in leng.items():
        ph["astrocytes (%s)" % st] = dict(G1=d["ntc_phase_shares"][0], S=d["ntc_phase_shares"][1],
                                          G2M=d["ntc_phase_shares"][2],
                                          mki67_or_top2a_positive=d["ntc_mki67_or_top2a_positive"])
    order = ["K562", "K562d8", "Jurkat", "RPE1", "HepG2", "HCT116", "HEK293T", "hESC",
             "CD4_D2rest", "CD4_D3rest", "CD4_D4rest", "astrocytes (vehicle)", "astrocytes (ITC)"]
    order = [o for o in order if o in ph]
    ax = fig.add_subplot(gs[0, :2]); clean(ax); panel(ax, "a", dx=-0.07)
    x = np.arange(len(order)); bot = np.zeros(len(order))
    for key, col in (("G1", "#dfe6e9"), ("S", "#b2bec3"), ("G2M", "#636e72")):
        v = np.array([ph[o][key] for o in order], float)
        ax.bar(x, v, 0.68, bottom=bot, color=col, edgecolor="white", lw=0.4, label=key)
        bot += v
    ax.axhline(0.6, color=RED, lw=0.8, ls="--")
    ax.text(len(order) - 0.4, 0.62, "G1 > 0.6: phase calls not used", color=RED, fontsize=6, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([o.replace("_", " ") for o in order], rotation=40, ha="right", fontsize=6)
    ax.set_ylabel("control-cell phase fraction"); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, fontsize=6.5)
    ax = fig.add_subplot(gs[0, 2]); clean(ax); panel(ax, "b", dx=-0.3)
    lab2, val = [], []
    for o in order:
        p = ph[o].get("proliferation_cp10k") or {}
        if p:
            lab2.append(o.replace("_", " ")); val.append(np.mean(list(p.values())))
    ax.barh(range(len(lab2)), val, 0.68, color=GREY)
    ax.set_yticks(range(len(lab2))); ax.set_yticklabels(lab2, fontsize=6)
    ax.set_xlabel("MKI67 / TOP2A in controls (CP10K)")
    ax.set_title("astrocytes (no pseudobulk CP10K available):\n%.0f%% and %.0f%% of control cells positive for\n"
                 "MKI67 or TOP2A, vehicle and cytokine"
                 % (100 * leng["vehicle"]["ntc_mki67_or_top2a_positive"],
                    100 * leng["ITC"]["ntc_mki67_or_top2a_positive"]), fontsize=6, color=GREY)
    fig.savefig(os.path.join(out, "figS1_phase_diagnostics.pdf"))
    fig.savefig(os.path.join(out, "figS1_phase_diagnostics.png"), dpi=300); plt.close(fig)
    print("wrote figS1", flush=True)


# ------------------------------------------------------------------ Figure S2
def figure_s2(A, S, out):
    fig = plt.figure(figsize=(183 * MM, 80 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.08, right=0.98, top=0.9, bottom=0.2)
    rows = S["classify_rows"]
    exp = np.array([r["expected_ctrl"] for r in rows], float)
    und = np.array([r["n_und"] / max(r["n_kd"] + r["n_esc"] + r["n_und"], 1) for r in rows], float)
    ax = fig.add_subplot(gs[0, 0]); clean(ax); panel(ax, "a", dx=-0.3)
    ax.scatter(exp, und, s=3, color=GREY, alpha=0.6, lw=0)
    ax.axvline(3, color=RED, lw=0.8, ls="--")
    ax.text(3.2, 0.95, "analysable: expected\ncount >= 3", color=RED, fontsize=6, va="top")
    ax.set_xscale("log"); ax.set_xlabel("expected target counts in controls")
    ax.set_ylabel("undetermined fraction"); ax.set_ylim(0, 1)
    ax = fig.add_subplot(gs[0, 1]); clean(ax); panel(ax, "b", dx=-0.3)
    esc = np.array([r["n_esc"] / max(r["n_esc"] + r["n_kd"], 1) for r in rows], float)
    res = np.array([r["resid"] for r in rows], float)
    ax.scatter(res, esc, s=3, color=ELIM, alpha=0.6, lw=0)
    ax.set_xlabel("survivor residual (pseudobulk)"); ax.set_ylabel("escaped fraction of classified cells")
    ax.set_title("classification agrees with the bulk residual\n(same counts; a self-check, not evidence)", fontsize=6.5)
    ax = fig.add_subplot(gs[0, 2]); clean(ax); panel(ax, "c", dx=-0.3)
    c = S["classify"]
    bars = [("silenced", c["Q5_mkd_median_ess"], BLUE), ("escaped", c["Q5_mkd_median_ess"] * c["Q1_esc_over_kd_median_ess"], ELIM),
            ("tail null", c["Q5_null_median_ess"], RED), ("thinning null", c["Q5p_null2_median_ess"], GREY),
            ("unthinned null", c["null3_median_ess"], ORANGE)]
    ax.bar(range(len(bars)), [b[1] for b in bars], 0.62, color=[b[2] for b in bars])
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], rotation=30, ha="right", fontsize=6)
    ax.set_yscale("log"); ax.set_ylabel("magnitude, most essential stratum")
    ax.set_title("selection controls", fontsize=7)
    fig.savefig(os.path.join(out, "figS2_classification_diagnostics.pdf"))
    fig.savefig(os.path.join(out, "figS2_classification_diagnostics.png"), dpi=300); plt.close(fig)
    print("wrote figS2", flush=True)


# ------------------------------------------------------------------ Figure S3
def figure_s3(A, B, out):
    fig = plt.figure(figsize=(183 * MM, 80 * MM))
    gs = fig.add_gridspec(1, 2, wspace=0.45, left=0.3, right=0.97, top=0.9, bottom=0.18)
    ci = A["ci"]
    items = [("hESC / K562, all targets", ci["bb"]["all_targets"], None),
             ("hESC / K562, knockdown matched", ci["bb"]["E2_both_good"], None),
             ("hESC normalised abundance, < -1", ci["bc"]["lt_-1"]["norm"],
              ci["bc"]["lt_-1"]["replicate_guide_medians"]),
             ("RPE1 composition share", ci["bm"]["RPE1"]["lt_-1"]["comp_share"], None),
             ("K562 composition share", ci["bm"]["K562"]["lt_-1"]["comp_share"],
              [ci["bv"]["lt_-1"]["comp_share"]["est"]]),
             ("hESC selection term", ci["bm"]["hESC"]["lt_-1"]["selection"], None),
             ("RPE1 drho, own composition", ci["br"]["RPE1_lt_-1"]["delta_own"], None),
             ("RPE1 drho, own minus mismatch", ci["br"]["RPE1_lt_-1"]["own_minus_same_tier"], None)]
    ax = fig.add_subplot(gs[0, 0]); clean(ax); panel(ax, "a", dx=-0.62)
    for i, (name, d, rep) in enumerate(items):
        e, lo, hi = fmt_ci(d)
        if e is None:
            continue
        sc = abs(e) if abs(e) > 0 else 1
        ax.errorbar([e / sc], [i], xerr=[[(e - lo) / sc], [(hi - e) / sc]], fmt="o", ms=3.4, color=BLUE,
                    elinewidth=0.8, capsize=1.8)
        if rep:
            for r in rep:
                if r:
                    ax.scatter([r / sc], [i], s=14, marker="|", color=ORANGE, zorder=3)
    ax.axvline(1, color=GREY, lw=0.6, ls=":")
    ax.set_yticks(range(len(items))); ax.set_yticklabels([n for n, _, _ in items], fontsize=6)
    ax.set_xlabel("value relative to its own point estimate")
    ax.set_xlim(0.2, 1.8)
    ax.scatter([], [], marker="|", s=20, color=ORANGE, label="replicate values")
    ax.errorbar([], [], xerr=[[0]], fmt="o", ms=3, color=BLUE, label="within-experiment interval")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6)
    ax = fig.add_subplot(gs[0, 1]); clean(ax); panel(ax, "b", dx=-0.25)
    pl = B["priming_maffa"]["per_line"]
    w = [(r["S_ci"][1] - r["S_ci"][0]) / 2 for r in pl]
    ax.hist(w, bins=10, color=GREY)
    ax.axvline(np.std([r["S"] for r in pl]), color=RED, lw=1.0)
    ax.text(np.std([r["S"] for r in pl]) * 1.03, ax.get_ylim()[1] * 0.9, "spread between\nlines", color=RED, fontsize=6)
    ax.set_xlabel("half-width of a line's own interval")
    ax.set_ylabel("cell lines")
    ax.set_title("when replicates exist, they bound the question\n(16 lines: spread < error, so no between-line claim)",
                 fontsize=6.5)
    fig.savefig(os.path.join(out, "figS3_intervals.pdf"))
    fig.savefig(os.path.join(out, "figS3_intervals.png"), dpi=300); plt.close(fig)
    print("wrote figS3", flush=True)


def main():
    fd, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    A = json.load(open(os.path.join(fd, "figdata_a100.json"), encoding="utf-8"))
    B = json.load(open(os.path.join(fd, "figdata_5090.json"), encoding="utf-8"))
    S = json.load(open(os.path.join(fd, "suppdata_a100.json"), encoding="utf-8"))
    S5 = json.load(open(os.path.join(fd, "suppdata_5090.json"), encoding="utf-8"))
    n1 = table_s1(A, B, out); n2 = table_s2(A, S, out); n3 = table_s3(A, B, out); n4 = table_s4(A, out)
    figure_s1(A, B, S, S5, out); figure_s2(A, S, out); figure_s3(A, B, out)
    print("rows: S1 %d, S2 %d, S3 %d, S4 %d" % (n1, n2, n3, n4), flush=True)


if __name__ == "__main__":
    main()
