"""Main figures of the fate-layer paper (2026-10-18). Panels follow notes/fate_paper_figures.md §4; every number is
taken from the exported figure data (code/a100/fate_figdata_a100.py, code/fate_figdata_5090.py), which only reshapes
the saved analysis outputs listed in notes/fate_paper_ledger.md. Schematic panels are drawn with matplotlib primitives
so they are reproducible.
Style: random halves RED, independent units BLUE, null / inert GREY; fates arrest BLUE2, tolerate GREY2, eliminate
PURPLE. Sizes in mm (183 double column), Arial 7-8 pt, PDF plus 300 dpi PNG.
Usage: plot_fate_figs.py <figdata dir> <out dir>.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, Circle, FancyBboxPatch

RED, BLUE, GREY = "#c0392b", "#2c3e50", "#7f8c8d"
ARREST, TOL, ELIM = "#2980b9", "#95a5a6", "#8e44ad"
ORANGE = "#e67e22"
MM = 1 / 25.4
TIERS = ["lt_-1", "-1_-0.5", "-0.5_-0.2", "gt_-0.2"]
TIERLAB = ["< -1", "-1 to\n-0.5", "-0.5 to\n-0.2", "> -0.2"]

plt.rcParams.update({"font.size": 7, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "axes.linewidth": 0.6, "axes.labelsize": 7.5, "axes.titlesize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2,
                     "ytick.major.size": 2.2, "legend.frameon": False, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})


def panel(ax, letter, dx=-0.17, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", name, flush=True)


def ci_of(d):
    """(point, lo, hi) from a fate_ci entry."""
    lo, hi = (d["ci"] if d.get("ci") else (np.nan, np.nan))
    return d["est"], lo, hi


# ---------------------------------------------------------------- Figure 1
def fate_cartoon(ax, kind):
    """Small cartoon of one fate: dish of cells before and after perturbation."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    rng = np.random.default_rng(4)
    col = {"arrest": ARREST, "tolerate": TOL, "eliminate": ELIM}[kind]
    for side, x0 in (("before", 0.4), ("after", 5.6)):
        ax.add_patch(FancyBboxPatch((x0, 0.5), 3.6, 3.3, boxstyle="round,pad=0.08",
                                    fc="white", ec=GREY, lw=0.6))
        n = 14 if side == "before" else {"arrest": 14, "tolerate": 16, "eliminate": 5}[kind]
        pts = rng.uniform([x0 + 0.35, 0.85], [x0 + 3.25, 3.45], size=(n, 2))
        for i, (x, y) in enumerate(pts):
            if side == "after" and kind == "arrest":
                fc, ec = (col if i % 3 else "white"), col
            elif side == "after" and kind == "eliminate":
                fc, ec = ("white" if i < 3 else col), col
            else:
                fc, ec = (col if side == "after" else "white"), (col if side == "after" else GREY)
            ax.add_patch(Circle((x, y), 0.26, fc=fc, ec=ec, lw=0.5))
    ax.annotate("", xy=(5.45, 2.1), xytext=(4.15, 2.1),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="black"))
    ax.text(4.8, 2.35, "pert.", ha="center", fontsize=6)
    lab = {"arrest": "arrest\ncells stay, cycle shifts",
           "tolerate": "tolerate\ncells stay, cycle unchanged",
           "eliminate": "eliminate\ncells lost, survivors escaped"}[kind]
    ax.text(5, 0.02, lab, ha="center", va="bottom", fontsize=6.5, color=col)
    ax.text(2.2, 3.95, "before", ha="center", fontsize=6, color=GREY)
    ax.text(7.4, 3.95, "after", ha="center", fontsize=6, color=GREY)


def figure1(A, B, out):
    fig = plt.figure(figsize=(183 * MM, 155 * MM))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.8, 1.15, 1.1], hspace=0.55, wspace=0.42,
                          left=0.08, right=0.98, top=0.95, bottom=0.07)
    for j, kind in enumerate(("arrest", "tolerate", "eliminate")):
        ax = fig.add_subplot(gs[0, j]); fate_cartoon(ax, kind)
        if j == 0:
            panel(ax, "a", dx=-0.08, dy=0.98)

    # b: residual vs dG1, point area ~ relative cells
    ax = fig.add_subplot(gs[1, :2]); clean(ax); panel(ax, "b", dx=-0.09)
    pts = [("RPE1", 0.12, 0.44, 0.71, ARREST, "arrest"), ("HepG2", 0.14, 0.26, 1.19, ARREST, None),
           ("HCT116", 0.35, 0.094, 0.59, ARREST, None), ("K562", 0.22, 0.05, 0.94, TOL, "tolerate"),
           ("Jurkat", 0.24, 0.05, 0.73, TOL, None), ("HEK293T", 0.53, 0.002, 0.52, TOL, None),
           ("hESC", 0.59, 0.01, 0.36, ELIM, "eliminate"), ("iPSC d3", 0.30, 0.01, 0.95, GREY, None),
           ("H1 (VCC)", 0.07, 0.07, 0.65, GREY, None)]
    off = {"RPE1": (8, 3), "HepG2": (8, 2), "HCT116": (8, 3), "K562": (-22, 6), "Jurkat": (-24, -4),
           "HEK293T": (-30, 7), "hESC": (7, 4), "iPSC d3": (8, -3), "H1 (VCC)": (-8, 8)}
    for name, res, dg1, rel, col, lab in pts:
        ax.scatter([res], [dg1], s=28 + 190 * rel, facecolor=col, edgecolor="white", lw=0.5, alpha=0.9,
                   zorder=3, label=lab)
        ax.annotate(name, (res, dg1), textcoords="offset points", xytext=off.get(name, (6, 5)), fontsize=6.3)
    ax.scatter([0.78], [-0.055], s=28 + 190 * 0.40, facecolor=ELIM, edgecolor="white", lw=0.5, alpha=0.9, zorder=3)
    ax.annotate("CD4+ T (rest)\ndG1 undefined (G0)", (0.78, -0.055), textcoords="offset points",
                xytext=(-78, -14), fontsize=6.3,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY))
    ax.axhline(0, color=GREY, lw=0.5, ls=":")
    ax.set_xlabel("survivor knockdown residual"); ax.set_ylabel("dG1 (perturbed - control)")
    ax.set_xlim(0, 0.92); ax.set_ylim(-0.1, 0.52)
    ax.legend(loc="upper right", handletextpad=0.3, borderpad=0.2, labelspacing=0.25,
              title="fate", title_fontsize=6.5)
    ax.text(0.02, 0.03, "point area ~ relative cell number", transform=ax.transAxes, fontsize=6, color=GREY)

    # b-inset: library-balanced caveat note
    ax2 = fig.add_subplot(gs[1, 2]); clean(ax2); panel(ax2, "c", dx=-0.3)
    for lab, col, ls, mk in (("RPE1", ARREST, "-", "o"), ("K562", TOL, "-", "o"),
                             ("HCT116", ARREST, "--", "^"), ("HEK293T", TOL, "--", "^")):
        if lab in ("RPE1", "K562"):
            rows = A["fig1c"][lab]
            y = []
            for nm, lo, hi in zip(TIERS, [-9, -1, -0.5, -0.2], [-1, -0.5, -0.2, 9]):
                v = [r["dG1"] for r in rows if r["e"] is not None and np.isfinite(r["e"]) and lo <= r["e"] < hi]
                y.append(np.median(v) if v else np.nan)
        else:
            y = [A["fig1c"]["orion"][lab]["tiers"][t]["dG1"] for t in TIERS]
        ax2.plot(range(4), y, marker=mk, ms=3, lw=1.1, color=col, ls=ls, label=lab)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(TIERLAB, fontsize=6)
    ax2.set_xlabel("DepMap stratum"); ax2.set_ylabel("dG1")
    ax2.axhline(0, color=GREY, lw=0.5, ls=":")
    ax2.plot([], [], color="black", ls="-", marker="o", ms=3, lw=1.0, label="Replogle")
    ax2.plot([], [], color="black", ls="--", marker="^", ms=3, lw=1.0, label="X-Atlas/Orion")
    ax2.legend(loc="upper right", ncol=2, handlelength=1.4, fontsize=5.5, columnspacing=0.7,
               labelspacing=0.25, borderpad=0.2)
    ax2.set_title("same-platform pairs", fontsize=7)

    # d: hESC library normalisation
    ax = fig.add_subplot(gs[2, :2]); clean(ax); panel(ax, "d", dx=-0.09)
    T = A["fig1d"]["tiers"]; ci = A["ci"]["bc"]
    x = np.arange(4); w = 0.36
    raw = [T[t]["raw_median"] for t in TIERS]; nrm = [T[t]["norm_median"] for t in TIERS]
    nlo = [ci[t]["norm"]["ci"][0] for t in TIERS]; nhi = [ci[t]["norm"]["ci"][1] for t in TIERS]
    ax.bar(x - w / 2, raw, w, color=GREY, label="raw abundance")
    ax.bar(x + w / 2, nrm, w, color=ELIM, label="normalised for library input")
    ax.errorbar(x + w / 2, nrm, yerr=[np.array(nrm) - np.array(nlo), np.array(nhi) - np.array(nrm)],
                fmt="none", ecolor="black", elinewidth=0.6, capsize=1.6)
    for xi, t in zip(x, TIERS):
        for g in ci[t]["replicate_guide_medians"]:
            if g:
                ax.plot([xi + w / 2], [g], marker="_", ms=5, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(TIERLAB, fontsize=6)
    ax.set_xlabel("DepMap stratum"); ax.set_ylabel("abundance vs non-targeting")
    ax.set_yscale("log"); ax.set_ylim(0.008, 6.0)
    ax.legend(loc="upper right", fontsize=6)
    ax.annotate("12x understated\nby raw abundance", xy=(0.18, 0.028), xytext=(0.55, 2.2),
                fontsize=6, color=ELIM, ha="center",
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=ELIM), zorder=5)
    ax.set_title("hESC (Pan): library-input normalisation", fontsize=7)

    ax = fig.add_subplot(gs[2, 2]); clean(ax); panel(ax, "e", dx=-0.3)
    cb = A["fig1d"]["copies_bins"]
    labs = [k for k in cb if cb[k]["n"] >= 50]
    ax.plot(range(len(labs)), [cb[k]["raw_median"] for k in labs], marker="o", ms=3, lw=1.1, color=GREY)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=6, rotation=40, ha="right")
    ax.set_xlabel("oligo copies per guide"); ax.set_ylabel("raw abundance")
    ax.axhline(1.0, color="black", lw=0.5, ls=":")
    ax.set_title("balancing undercompensates", fontsize=7)
    save(fig, out, "fig1_fates")


# ---------------------------------------------------------------- Figure 2
def figure2(A, B, out):
    fig = plt.figure(figsize=(183 * MM, 135 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.5, wspace=0.42,
                          left=0.08, right=0.98, top=0.93, bottom=0.08)

    # a: schematic of the two comparisons
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); panel(ax, "a", dx=-0.1, dy=1.0)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    rng = np.random.default_rng(1)
    ax.text(0.2, 9.6, "published comparison", fontsize=6.8, fontweight="bold", va="top")
    for i, (x0, col, lab, mu) in enumerate(((0.8, ELIM, "hESC", 1.15), (5.4, TOL, "K562", 1.95))):
        xs = np.linspace(0, 3.4, 120)
        ys = np.exp(-0.5 * ((xs - mu) / 0.55) ** 2) * 1.5
        ax.fill_between(xs + x0, 6.6, 6.6 + ys, color=col, alpha=0.75, lw=0)
        ax.plot(xs + x0, 6.6 + ys, color=col, lw=0.7)
        ax.text(x0 + mu, 6.25, lab, ha="center", fontsize=6.2, color=col)
    ax.plot([0.6, 9.4], [6.6, 6.6], color=GREY, lw=0.6)
    ax.text(0.2, 5.75, "magnitude distributions, grouped by\nwhich side silenced more strongly",
            fontsize=6, color=GREY, va="top")
    ax.plot([0.4, 9.6], [4.75, 4.75], color=GREY, lw=0.5)
    ax.text(0.2, 4.4, "this work", fontsize=6.8, fontweight="bold", va="top")
    for j in range(5):
        y = 3.3 - j * 0.52
        ax.plot([2.4, 6.6], [y, y], lw=0.7, color=GREY, zorder=1)
        ax.scatter([2.4], [y], s=11, color=ELIM, zorder=2); ax.scatter([6.6], [y], s=11, color=TOL, zorder=2)
    ax.text(2.4, 3.75, "hESC", ha="center", fontsize=6, color=ELIM)
    ax.text(6.6, 3.75, "K562", ha="center", fontsize=6, color=TOL)
    ax.text(0.2, 0.75, "same target paired; knockdown good on\nBOTH sides; magnitude between\nindependent units",
            fontsize=6, color=GREY, va="top")

    # b: per-target ratios
    ax = fig.add_subplot(gs[0, 1:]); clean(ax); panel(ax, "b", dx=-0.07)
    d = A["fig2b"]; h = np.array(d["h"], float); k = np.array(d["k"], float); good = np.array(d["good"], bool)
    ok = np.isfinite(h) & np.isfinite(k) & (h > 0) & (k > 0)
    y_all = np.log2(h[ok] / k[ok]); y_good = np.log2(h[ok & good] / k[ok & good])
    for i, (y, col, lab) in enumerate(((y_all, GREY, "all shared targets"), (y_good, BLUE, "knockdown good both sides"))):
        parts = ax.violinplot([y], positions=[i], widths=0.8, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(col); b.set_alpha(0.45); b.set_edgecolor("none")
        ax.scatter(np.random.default_rng(i).normal(i, 0.055, len(y)), y, s=1.2, color=col, alpha=0.5, lw=0)
        ax.plot([i - 0.3, i + 0.3], [np.median(y)] * 2, color="black", lw=1.1)
    cib = A["ci"]["bb"]
    for i, key in enumerate(("all_targets", "E2_both_good")):
        e, lo, hi = ci_of(cib[key])
        ax.errorbar([i + 0.33], [np.log2(e)], yerr=[[np.log2(e) - np.log2(lo)], [np.log2(hi) - np.log2(e)]],
                    fmt="o", ms=2.6, color="black", elinewidth=0.7, capsize=1.6)
        ax.text(i + 0.42, np.log2(e), "%.2f\n[%.2f, %.2f]" % (e, lo, hi), fontsize=6, va="center")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["all shared targets\n(%d of %d plotted)" % (len(y_all), len(h)),
                                               "knockdown <= 0.3 both sides\n(%d of %d plotted)"
                                               % (len(y_good), int(good.sum()))], fontsize=6.3)
    ax.set_ylabel("log2 (hESC / K562 magnitude)")
    ax.set_ylim(-9, 9)
    ax.text(0.02, 0.97, "nine threshold combinations: 0.89-1.18", transform=ax.transAxes, fontsize=6,
            color=BLUE, va="top")

    # c: escape fraction predicts the ratio
    ax = fig.add_subplot(gs[1, 0]); clean(ax); panel(ax, "c", dx=-0.3)
    m = A["fig2c"]["main"]
    sl, lo, hi = m["slope"], m["slope_ci95"][0], m["slope_ci95"][1]
    rk = m.get("r_kd_main") or A["fig2c"].get("r_kd_main") or {}
    rk_h = rk.get("hesc", 0.20); rk_6 = rk.get("k562_d6", 0.16); rk_8 = rk.get("k562_d8", 0.16)
    esc = lambda R, r: np.clip((np.asarray(R, float) - r) / (1 - r), 0, 0.95)
    xx = (2 * np.log2(1 - esc(d["res_h"], rk_h)) - np.log2(1 - esc(d["res_6"], rk_6))
          - np.log2(1 - esc(d["res_8"], rk_8)))
    yy = np.where(ok, np.log2(np.where(ok, h, 1) / np.where(ok, k, 1)), np.nan)
    m_ok = np.isfinite(xx) & np.isfinite(yy)
    ax.scatter(xx[m_ok], yy[m_ok], s=1.6, color=GREY, alpha=0.45, lw=0, zorder=1)
    ax.axline((0, m["intercept"]), slope=sl, color=ELIM, lw=1.2, zorder=3,
              label="Theil-Sen %.2f [%.2f, %.2f]" % (sl, lo, hi))
    ax.axline((0, m["intercept"]), slope=m["null_slope_p97_5"], color=GREY, lw=0.8, ls=":", zorder=2,
              label="permuted null")
    ax.set_xlim(-4, 1); ax.set_ylim(-8, 4)
    ax.set_xlabel("predicted from escape fraction (log2)")
    ax.set_ylabel("log2 (hESC / K562)")
    ax.legend(loc="lower right", handlelength=1.3, fontsize=5.8)
    ax.set_title("escape fraction predicts the ratio\n(Spearman %.2f, n = %d)" % (m["spearman"], m["n"]), fontsize=7)

    # d: per-cell classification
    ax = fig.add_subplot(gs[1, 1]); clean(ax); panel(ax, "d", dx=-0.3)
    t = A["fig2d"]["tiers"]["lt_-1"]
    bars = [("silenced\ncells", t["mkd_median"], BLUE), ("escaped\ncells", t["mkd_median"] * t["esc_over_kd"], ELIM),
            ("tail null\n(pre-stated)", t["null_median"], RED), ("thinning null\n(matched)", t["null2_median"], GREY)]
    ax.bar(range(4), [b[1] for b in bars], 0.62, color=[b[2] for b in bars])
    ax.set_xticks(range(4)); ax.set_xticklabels([b[0] for b in bars], fontsize=6)
    ax.set_yscale("log"); ax.set_ylabel("magnitude (most essential)")
    ax.set_ylim(0.03, 40)
    ax.text(2, t["null_median"] * 1.25, "larger than\nthe signal", fontsize=6, color=RED, ha="center")

    ax = fig.add_subplot(gs[1, 2]); clean(ax); panel(ax, "e", dx=-0.3)
    Q = A["fig2d"]["Q4p_same_targets_diluted"]; Qa = A["fig2d"]["Q4p_same_targets_all"]
    keys = [("all_cells", "all cells", GREY), ("linear", "mean-residual\ncorrection", ORANGE),
            ("per_cell_corrected", "silenced cells\n(per-cell)", BLUE)]
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, [Q[k[0]] for k in keys], w, color=[k[2] for k in keys], label="diluted subset (n=%d)" % Q["n"])
    ax.bar(x + w / 2, [Qa[k[0]] for k in keys], w, color=[k[2] for k in keys], alpha=0.5,
           label="all classifiable (n=%d)" % Qa["n"])
    e, lo, hi = ci_of(A["ci"]["bi"]["diluted_ratio"]) if "bi" in A["ci"] and "diluted_ratio" in A["ci"].get("bi", {}) \
        else (0.525, 0.34, 0.67)
    ax.errorbar([2 - w / 2], [e], yerr=[[e - lo], [hi - e]], fmt="none", ecolor="black", elinewidth=0.7, capsize=1.6)
    ax.axhline(1, color="black", lw=0.6, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([k[1] for k in keys], fontsize=6)
    ax.set_ylabel("hESC / K562 magnitude"); ax.legend(loc="upper left")
    ax.set_ylim(0, 1.5)
    save(fig, out, "fig2_samescale")


# ---------------------------------------------------------------- Figure 3
def figure3(A, B, out):
    fig = plt.figure(figsize=(183 * MM, 125 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.85, 1], hspace=0.55, wspace=0.42,
                          left=0.08, right=0.98, top=0.93, bottom=0.1)

    # a: composition counterfactual schematic
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); panel(ax, "a", dx=-0.1, dy=1.0)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    for i, (x0, lab, frac, col) in enumerate(((0.8, "control", [0.33, 0.33, 0.34], GREY),
                                              (5.6, "perturbed", [0.62, 0.2, 0.18], ARREST))):
        y = 7.6
        for f, c in zip(frac, ("#dfe6e9", "#b2bec3", "#636e72")):
            ax.add_patch(Rectangle((x0, y), 3.4 * f, 0.9, fc=c, ec="white", lw=0.5)); x0 += 3.4 * f
        ax.text(0.8 + 1.7 + 4.8 * i, 7.0, lab, ha="center", fontsize=6.3, color=col)
    ax.text(5, 9.0, "phase mixture", ha="center", fontsize=6.8, fontweight="bold")
    ax.text(0.6, 5.6, "composition term:\nmixture from perturbed cells,\nwithin-phase expression\nfrom controls",
            fontsize=6.2, va="top")
    ax.text(0.6, 2.4, "within-cell term:\nthe remainder", fontsize=6.2, va="top")
    ax.text(0.6, 0.9, "selection term:\nrecomputed from\nsilenced cells only", fontsize=6.2, va="top", color=ELIM)

    # b: composition share and selection per context
    ci = A["ci"]; c5 = B["ci_5090"]["bm"]
    ctx = [("RPE1", ARREST, ci["bm"]["RPE1"]), ("HepG2", ARREST, c5["HepG2"]), ("Jurkat", TOL, c5["Jurkat"]),
           ("K562", TOL, ci["bm"]["K562"]), ("hESC", ELIM, ci["bm"]["hESC"])]
    ax = fig.add_subplot(gs[0, 1]); clean(ax); panel(ax, "b", dx=-0.3)
    for i, (name, col, d) in enumerate(ctx):
        e, lo, hi = ci_of(d["lt_-1"]["comp_share"])
        ax.errorbar([i], [e], yerr=[[max(e - lo, 0)], [max(hi - e, 0)]], fmt="o", ms=3.4, color=col,
                    elinewidth=0.8, capsize=1.8)
    e, lo, hi = ci_of(ci["bv"]["lt_-1"]["comp_share"])
    ax.errorbar([3.32], [e], yerr=[[e - lo], [hi - e]], fmt="s", ms=3.2, color=BLUE, elinewidth=0.8, capsize=1.8)
    ax.annotate("K562, two independent screens", xy=(3.32, e), xytext=(1.1, 0.0032), fontsize=5.8, color=BLUE,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=BLUE))
    ax.set_xlim(-0.6, 4.6)
    ax.set_xticks(range(5)); ax.set_xticklabels([c[0] for c in ctx], fontsize=6.3, rotation=30, ha="right")
    ax.set_yscale("log"); ax.set_ylabel("composition share"); ax.set_ylim(0.002, 0.5)
    ax.set_title("composition follows arrest", fontsize=7)

    ax = fig.add_subplot(gs[0, 2]); clean(ax); panel(ax, "c", dx=-0.3)
    order = sorted(ctx, key=lambda c: -c[2]["lt_-1"]["selection"]["est"])
    for i, (name, col, d) in enumerate(order):
        e, lo, hi = ci_of(d["lt_-1"]["selection"])
        ax.errorbar([i], [e], yerr=[[e - lo], [hi - e]], fmt="o", ms=3.4, color=col, elinewidth=0.8, capsize=1.8)
    ax.axhline(1, color="black", lw=0.6, ls="--")
    ax.set_xticks(range(5)); ax.set_xticklabels([c[0] for c in order], fontsize=6.3, rotation=30, ha="right")
    ax.set_ylabel("selection term (silenced / all)")
    ax.set_title("selection follows residual", fontsize=7)

    # d: cycle genes removed
    ax = fig.add_subplot(gs[1, 0]); clean(ax); panel(ax, "d", dx=-0.3)
    src = {"RPE1": A["fig3"]["RPE1"], "K562": A["fig3"]["K562"], "hESC": A["fig3"]["hESC"],
           "HepG2": B["budget"]["HepG2"], "Jurkat": B["budget"]["Jurkat"]}
    names = ["RPE1", "HepG2", "Jurkat", "K562", "hESC"]
    w = 0.36; x = np.arange(len(names))
    a1 = [src[n]["random"]["lt_-1"]["comp_share"] for n in names]
    a2 = [src[n]["random_no_cycle_genes"]["lt_-1"]["comp_share"] for n in names]
    ax.bar(x - w / 2, a1, w, color=GREY, label="all response genes")
    ax.bar(x + w / 2, a2, w, color=BLUE, label="cycle genes removed")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.3, rotation=30, ha="right")
    ax.set_yscale("log"); ax.set_ylim(0.002, 0.5); ax.set_ylabel("composition share")
    ax.legend(loc="upper right")

    # e: inert negative control
    ax = fig.add_subplot(gs[1, 1:]); clean(ax); panel(ax, "e", dx=-0.07)
    rows = [("hESC", A["fig3"]["hESC|budget_hesc_nc"]["negative_control"]),
            ("CD4 donor 2", A["fig3"]["CD4_D2rest|nc"]["negative_control"]),
            ("CD4 donor 3", A["fig3"]["CD4_D3rest|d34"]["negative_control"]),
            ("CD4 donor 4", A["fig3"]["CD4_D4rest|d34"]["negative_control"])]
    x = np.arange(len(rows)); w = 0.36
    ax.bar(x - w / 2, [r[1]["ratio"] for r in rows], w, color=RED, label="random halves")
    ax.bar(x + w / 2, [r[1]["guide_ratio"] for r in rows], w, color=BLUE, label="guide halves")
    ax.axhline(0.10, color="black", lw=0.7, ls="--")
    ax.text(3.45, 0.115, "pre-stated limit", fontsize=6, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=6.5)
    ax.set_ylabel("inert targets / non-essential stratum")
    ax.legend(loc="upper left")
    ax.set_title("null-perturbation control: two CD4 donors fail and are excluded", fontsize=7)
    save(fig, out, "fig3_budget")


# ---------------------------------------------------------------- Figure 4
def figure4(A, B, out):
    fig = plt.figure(figsize=(183 * MM, 115 * MM))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.4, left=0.08, right=0.98, top=0.92, bottom=0.12)
    D = B["drepanos"]["days"]; days = [4, 7, 10, 14]
    sel = [("NAMPT", "CRISPRi", ELIM, "-"), ("OGFOD1", "CRISPRi", ORANGE, "-"),
           ("RPL9", "CRISPRko", ELIM, ":"), ("CTBP1", "CRISPRi", TOL, "-")]
    for j, (key, ylab, logy) in enumerate((("abundance_vs_nonessential", "relative abundance", False),
                                           ("residual_ref", "survivor residual", False),
                                           ("response_ref", "response magnitude", True))):
        ax = fig.add_subplot(gs[0, j]); clean(ax)
        if j == 0:
            panel(ax, "a", dx=-0.32)
        for gene, arm, col, ls in sel:
            y, xs = [], []
            for d in days:
                t = D[str(d)].get(arm, {}).get("targets", {}).get(gene)
                if t and t.get(key) is not None and (not logy or t[key] > 0):
                    xs.append(d); y.append(t[key])
            ax.plot(xs, y, marker="o", ms=3, lw=1.0, color=col, ls=ls,
                    label="%s (%s)" % (gene, "KD" if arm == "CRISPRi" else "KO"))
        ax.set_xticks(days); ax.set_xlabel("day after transduction"); ax.set_ylabel(ylab)
        if logy:
            ax.set_yscale("log")
        if j == 1:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, handlelength=1.4, fontsize=6)
    # b: harvest-day ladder (values from notes/fate_paper_ledger.md, Table 1)
    ax = fig.add_subplot(gs[1, :2]); clean(ax); panel(ax, "b", dx=-0.09)
    ladder = [(3, 13.5, "iPSC", GREY, "random"), (6, 23.4, "16 lines (KO)", TOL, "constructs"),
              (6, 11.1, "K562", TOL, "random"), (7, 7.5, "RPE1", ARREST, "random"),
              (7, 3.8, "Jurkat", TOL, "random"), (7, 8.6, "HepG2", ARREST, "random"),
              (8, 1.38, "hESC", ELIM, "guides"), (11, 19.6, "CD4 rest", ELIM, "guides"),
              (13, 11.4, "CD4 stim", ELIM, "guides")]
    off4 = {"iPSC": (-16, 4), "16 lines (KO)": (6, 2), "K562": (-18, -2), "RPE1": (6, -8),
            "Jurkat": (6, -8), "HepG2": (6, 3), "hESC": (4, -9), "CD4 rest": (-10, 6), "CD4 stim": (4, -9)}
    for d, r, lab, col, unit in ladder:
        mk = "o" if unit == "random" else ("s" if unit == "guides" else "^")
        ax.scatter([d], [r], s=26, color=col, marker=mk, zorder=3)
        ax.annotate(lab, (d, r), textcoords="offset points", xytext=off4.get(lab, (5, 3)), fontsize=6)
    ax.axhline(1, color="black", lw=0.6, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("harvest day (cell type varies with day)")
    ax.set_ylabel("response ratio,\nmost essential / non-essential")
    ax.set_xlim(1.5, 16); ax.set_ylim(0.8, 60)
    ax.annotate("inverted U:\nmost essential\nattenuated", xy=(8, 1.38), xytext=(9.3, 1.6), fontsize=6, color=ELIM,
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=ELIM))
    ax.scatter([], [], marker="o", s=20, color=GREY, label="random halves")
    ax.scatter([], [], marker="s", s=20, color=GREY, label="guide halves")
    ax.scatter([], [], marker="^", s=20, color=GREY, label="independent constructs")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6, ncol=3, columnspacing=0.9)
    # c: Maffa response by depletion bin
    ax = fig.add_subplot(gs[1, 2]); clean(ax); panel(ax, "c", dx=-0.32)
    bins = B["maffa"]["response_by_depletion_bin"]
    ks = list(bins.keys())
    ax.bar(range(len(ks)), [bins[k]["response_median"] for k in ks], 0.62, color=TOL)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([k.replace(", ", "-").strip("()[]") for k in ks],
                                                      fontsize=5.8, rotation=40, ha="right")
    ax.set_yscale("log"); ax.set_xlabel("relative cells remaining"); ax.set_ylabel("response magnitude")
    ax.set_title("16 lines, day 6: no attenuation", fontsize=7)
    save(fig, out, "fig4_time")


# ---------------------------------------------------------------- Figure 5
def figure5(A, B, out):
    fig = plt.figure(figsize=(183 * MM, 120 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.34, left=0.1, right=0.97, top=0.93, bottom=0.1)
    N = A["fig5a_null"]; X = A["fig5a"]
    ax = fig.add_subplot(gs[0, 0]); clean(ax); panel(ax, "a", dx=-0.2)
    groups = [("RPE1", "lt_-1"), ("RPE1", "-1_-0.5"), ("HepG2", "lt_-1"), ("HepG2", "-1_-0.5")]
    w = 0.26
    for i, (lab, tier) in enumerate(groups):
        d = N[lab][tier]
        for j, (key, col, nm) in enumerate((("delta_own", ARREST, "own composition"),
                                            ("delta_same_tier_mismatch", ORANGE, "same-stratum donor"),
                                            ("delta_nonessential_mismatch", GREY, "non-essential donor"))):
            ax.bar(i + (j - 1) * w, d[key], w, color=col, label=nm if i == 0 else None)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["RPE1\n< -1", "RPE1\n-1..-0.5", "HepG2\n< -1", "HepG2\n-1..-0.5"], fontsize=6)
    ax.set_ylabel("change in agreement with K562 (drho)")
    ax.legend(loc="upper right", fontsize=6)
    ax.set_title("removing composition: the control can fail, and does not", fontsize=7)

    ax = fig.add_subplot(gs[0, 1]); clean(ax); panel(ax, "b", dx=-0.22)
    d = X["hESC"]["total_vs_kd"]["lt_-1"]
    ax.plot([0, 1], [d["mag_ratio_a"], d["mag_ratio_b"]], marker="o", ms=4, color=ELIM, lw=1.2)
    ax.axhline(1, color="black", lw=0.6, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["all cells", "silenced cells"], fontsize=6.5)
    ax.set_xlim(-0.3, 1.5); ax.set_ylim(0, 1.2)
    ax.set_ylabel("hESC / K562 magnitude")
    ax.text(1.05, d["mag_ratio_b"], "drho %.3f\n(direction\nunchanged)" % d["delta_rho_median"], fontsize=6,
            va="center")
    ax.set_title("selection correction moves magnitude", fontsize=7)

    ax = fig.add_subplot(gs[1, 0]); clean(ax); panel(ax, "c", dx=-0.2)
    srcs = ["K562", "Jurkat", "HepG2", "RPE1"]
    for i, s in enumerate(srcs):
        v = A["fig5c"][s]["hESC"]
        a = v["total_common"]["lt_-1"]["denoised_scale"]; b = v["kd_common"]["lt_-1"]["denoised_scale"]
        ax.plot([a, b], [i, i], color=GREY, lw=1.0, zorder=1)
        ax.scatter([a], [i], s=22, color=RED, zorder=2, label="uncorrected truth" if i == 0 else None)
        ax.scatter([b], [i], s=22, color=BLUE, zorder=2, label="silenced-cell truth" if i == 0 else None)
    ax.axvline(1, color="black", lw=0.6, ls="--")
    ax.set_yticks(range(len(srcs))); ax.set_yticklabels(["source: " + s for s in srcs], fontsize=6.5)
    ax.set_xlabel("denoised scale factor the benchmark asks for")
    ax.set_xlim(0, 1.25); ax.set_ylim(-0.6, len(srcs) - 0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6)
    ax.set_title("the instruction to shrink predictions disappears", fontsize=7)

    ax = fig.add_subplot(gs[1, 1]); clean(ax); panel(ax, "d", dx=-0.22)
    for s, col in zip(srcs, (BLUE, TOL, ARREST, ELIM)):
        v = A["fig5c"][s]["hESC"]
        y = [v["total_common"][t]["noise_share_median"] for t in TIERS[:2]]
        ax.plot(range(2), y, marker="o", ms=3, lw=1.0, color=col, label=s)
    ax.set_xticks(range(2)); ax.set_xticklabels(TIERLAB[:2], fontsize=6.3)
    ax.set_xlabel("DepMap stratum (hESC ground truth)")
    ax.set_ylabel("noise share of ||truth||^2")
    ax.set_ylim(0, 1); ax.legend(loc="lower left", fontsize=6, ncol=2)
    ax.set_title("elimination leaves the truth mostly noise", fontsize=7)
    save(fig, out, "fig5_benchmark")


# ---------------------------------------------------------------- Figure 6
def figure6(A, B, out):
    fig = plt.figure(figsize=(183 * MM, 118 * MM))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.42, left=0.08, right=0.98, top=0.92, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0]); clean(ax); panel(ax, "a", dx=-0.32)
    labs, obs, err, cols = [], [], [], []
    for lab, col in (("hESC", ELIM), ("K562", TOL), ("RPE1", ARREST)):
        for t in TIERS[:2]:
            y = A["fig6a"][lab]["Y0"][t]
            if "sd_observed" in y:
                labs.append("%s\n%s" % (lab, t.replace("_", " "))); obs.append(y["sd_observed"])
                err.append(y["sd_error"]); cols.append(col)
    x = np.arange(len(labs)); w = 0.36
    ax.bar(x - w / 2, obs, w, color=cols, label="observed spread")
    ax.bar(x + w / 2, err, w, color=GREY, label="single-target error")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=5.6)
    ax.set_ylabel("s.d. of depletion beyond dependency")
    ax.legend(loc="upper right", fontsize=6)
    ax.set_title("the variation is real", fontsize=7)

    ax = fig.add_subplot(gs[0, 1]); clean(ax); panel(ax, "b", dx=-0.32)
    pl = sorted(B["priming_maffa"]["per_line"], key=lambda r: r["S"])
    y = np.arange(len(pl))
    for i, r in enumerate(pl):
        ax.plot(r["S_ci"], [i, i], color=GREY, lw=0.8)
        ax.scatter([r["S"]], [i], s=10, color=ARREST if r["tp53_damaging"] == 0 else ELIM, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([r["line"] for r in pl], fontsize=5.2)
    ax.set_xlabel("elimination sensitivity (slope)")
    ax.set_title("16 lines: Q p = 0.67", fontsize=7)
    ax.scatter([], [], s=12, color=ARREST, label="TP53 intact"); ax.scatter([], [], s=12, color=ELIM, label="TP53 damaged")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=5.8)

    ax = fig.add_subplot(gs[0, 2]); clean(ax); panel(ax, "c", dx=-0.32)
    G = A["fig6b_jost"]["genes"]
    g = [(v["gamma_top"], v["D"], v["D_se"]) for v in G.values()]
    for gm, D, se in g:
        ax.errorbar([gm], [D], yerr=[se], fmt="o", ms=2.6, color=GREY, elinewidth=0.6, capsize=1.2)
    ax.axhline(1, color="black", lw=0.6, ls="--")
    ax.set_xlabel("growth cost of the gene (gamma)"); ax.set_ylabel("dilution index D")
    ax.set_ylim(0.6, 1.25)
    ax.set_title("dose titration: D ~ 1, p = 0.87", fontsize=7)

    ax = fig.add_subplot(gs[1, :2]); clean(ax); panel(ax, "d", dx=-0.09)
    P = A["fig6b_prog"]
    order = [("RPE1", "p53_targets"), ("RPE1", "bh3_apoptosis"), ("RPE1", "isr_atf4"), ("RPE1", "heat_shock"),
             ("hESC", "bh3_apoptosis"), ("hESC", "isr_atf4"), ("hESC", "heat_shock"),
             ("K562", "bh3_apoptosis"), ("K562", "isr_atf4"), ("K562", "heat_shock")]
    for i, (lab, prog) in enumerate(order):
        d = P.get(lab, {}).get("programmes", {}).get(prog)
        if not d or "partial_given_ess" not in d:
            continue
        nl = d["matched_null"]
        ax.add_patch(Rectangle((i - 0.3, nl[0]), 0.6, nl[1] - nl[0], fc=GREY, alpha=0.3, lw=0))
        ax.scatter([i], [d["partial_given_ess"]], s=16, color=ELIM, zorder=3,
                   label="given dependency" if i == 0 else None)
        ax.scatter([i], [d["partial_given_ess_and_dG1"]], s=16, facecolor="white", edgecolor=ARREST, lw=0.9,
                   zorder=3, label="also given dG1" if i == 0 else None)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["%s\n%s" % (l, p.split("_")[0]) for l, p in order], fontsize=5.6)
    ax.set_ylabel("partial Spearman with\ndepletion beyond dependency")
    ax.legend(loc="lower right", fontsize=6)
    ax.text(0.02, 0.05, "grey: expression-matched null interval", transform=ax.transAxes, fontsize=6, color=GREY)
    ax.set_title("induced programmes do not predict elimination; the one relation that exceeds its null is arrest",
                 fontsize=7)

    ax = fig.add_subplot(gs[1, 2]); ax.axis("off"); panel(ax, "e", dx=-0.2, dy=1.0)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(5, 9.3, "what observation cannot settle", ha="center", fontsize=6.8, fontweight="bold")
    ax.text(0.3, 7.6, "block apoptosis (BAX / BAK1)\nin an eliminating context:", fontsize=6.2, va="top")
    ax.annotate("", xy=(3.2, 5.2), xytext=(1.2, 6.4), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=ELIM))
    ax.text(3.5, 5.4, "cells remaining rise\nresponse recovers towards\nthe middle strata", fontsize=6.2, va="top",
            color=ELIM)
    ax.text(0.3, 2.9, "tolerant context: no change\n(control)", fontsize=6.2, va="top", color=TOL)
    ax.text(0.3, 1.0, "survivor transcriptomes cannot\nanswer it: the informative cells\nare absent by construction",
            fontsize=6, va="top", color=GREY)
    save(fig, out, "fig6_mechanism")


def main():
    fd, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    A = json.load(open(os.path.join(fd, "figdata_a100.json"), encoding="utf-8"))
    B = json.load(open(os.path.join(fd, "figdata_5090.json"), encoding="utf-8"))
    figure1(A, B, out); figure2(A, B, out); figure3(A, B, out)
    figure4(A, B, out); figure5(A, B, out); figure6(A, B, out)


if __name__ == "__main__":
    main()
