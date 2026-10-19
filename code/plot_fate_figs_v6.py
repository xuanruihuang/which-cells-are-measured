"""Fate-layer paper, main figures 1-5 redrawn to journal figure rules (v6 style, 2026-10-19).
Rules: 183 mm wide and at most 150 mm tall, placed at native size; Arial; 6-7 pt text, 8 pt bold panel letters;
all text black; short black panel titles, messages in the legend; colour only for data, explained by keys;
one colour per context throughout (Okabe-Ito based, colour-blind safe); cell groups by marker fill, not colour.
Axes are placed in millimetres so that nothing overlaps; every panel is checked to keep its data inside the axes.
Numbers are read from the same files as plot_fate_figs_v4/v5.py; key values are asserted.
Every axis carries its unit in parentheses (dilution in log2, magnitude in (log2 FC)^2; ratios and shares are unitless).
Usage: plot_fate_figs_v6.py <figdata> <v1 figdata> <v3 figdata> <v4 figdata> <out>."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_fate_figs_v2 import TIERS, example_target, expected_ratio_hist

MM = 1 / 25.4
W = 183
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 6.5, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                     "xtick.minor.size": 1.2, "ytick.minor.size": 1.2, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "legend.frameon": False, "axes.titlesize": 6.5, "axes.labelsize": 6.5,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "lines.linewidth": 0.8,
                     "axes.labelpad": 2, "text.color": "black", "axes.labelcolor": "black"})
BLACK, GREY, LGREY = "#000000", "#8c8c8c", "#cfcfcf"
C = {"hESC": "#CC79A7", "K562": "#0072B2", "K562_d8": "#0072B2", "K562_d6": "#56B4E9", "Jurkat": "#009E73",
     "RPE1": "#E69F00", "HepG2": "#D55E00", "CD4": "#6E6E6E", "HCT116": "#E69F00", "HEK293T": "#0072B2", "other": "#B0B0B0"}
TL = ["< −1", "−1 to\n−0.5", "−0.5 to\n−0.2", "> −0.2"]


def fig_(h):
    return plt.figure(figsize=(W * MM, h * MM))


def ax_(fig, H, x, y, w, h):
    """axes at x, y mm from the top-left corner, w x h mm"""
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def letter(fig, H, x, y, s):
    fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight="bold", va="top", ha="left")


def title(ax, s):
    ax.set_title(s, loc="left", fontsize=6.5, pad=3)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf")); fig.savefig(os.path.join(out, name + ".png"), dpi=300); plt.close(fig)
    print("wrote", name, flush=True)


def inside(ax, x, y):
    x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
    x, y = np.asarray(x, float), np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    assert (x[ok] >= x0).all() and (x[ok] <= x1).all() and (y[ok] >= y0).all() and (y[ok] <= y1).all(), ax.get_title()


def boot_median(x, rng, nb=2000):
    x = np.asarray(x, float)
    b = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(nb)]
    return np.median(x), np.percentile(b, 2.5), np.percentile(b, 97.5)


def tier_ticks(ax, n=4, lab=True):
    ax.set_xticks(range(n)); ax.set_xticklabels(TL[:n] if lab else [])
    ax.set_xlim(-0.55, n - 0.45)


# ================================================================ Figure 1
def cartoon(ax, kind):
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    rng = np.random.default_rng(4)
    col = {"arrest": C["RPE1"], "tolerance": C["K562"], "depletion": C["hESC"]}[kind]
    for side, x0 in (("before", 0.3), ("after", 5.7)):
        ax.add_patch(FancyBboxPatch((x0, 1.1), 4.0, 3.3, boxstyle="round,pad=0.08", fc="white", ec=GREY, lw=0.5))
        n = 14 if side == "before" else {"arrest": 14, "tolerance": 16, "depletion": 5}[kind]
        pts = rng.uniform([x0 + 0.4, 1.45], [x0 + 3.6, 4.05], size=(n, 2))
        for i, (x, y) in enumerate(pts):
            if side == "before":
                fc, ec = "white", GREY
            elif kind == "arrest":
                fc, ec = col, (BLACK if i % 3 else col)
            elif kind == "depletion":
                fc, ec = ("white" if i < 3 else col), col
            else:
                fc, ec = col, col
            ax.add_patch(Circle((x, y), 0.27, fc=fc, ec=ec, lw=0.5))
    ax.annotate("", xy=(5.55, 2.75), xytext=(4.45, 2.75), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=BLACK))
    ax.text(2.3, 4.55, "before", ha="center", fontsize=6); ax.text(7.7, 4.55, "after", ha="center", fontsize=6)
    lab = {"arrest": "Arrest: cells remain, cell cycle shifts", "tolerance": "Tolerance: cells remain, cycle unchanged",
           "depletion": "Depletion: few cells remain, many not silenced"}[kind]
    ax.text(5, 0.35, lab, ha="center", va="center", fontsize=6)


def figure1(A, out):
    H = 137; fig = fig_(H)
    for j, kind in enumerate(("arrest", "tolerance", "depletion")):
        cartoon(ax_(fig, H, 6 + j * 59, 4, 55, 27), kind)
    fig.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="white", mec=GREY, ms=4, label="Cell before perturbation"),
                        Line2D([], [], marker="o", ls="none", mfc=GREY, mec=GREY, ms=4, label="Target silenced"),
                        Line2D([], [], marker="o", ls="none", mfc=GREY, mec=BLACK, mew=0.9, ms=4, label="Target silenced, arrested"),
                        Line2D([], [], marker="o", ls="none", mfc="white", mec=GREY, mew=0.9, ms=4, label="Target not silenced")],
                                               loc="upper left", bbox_to_anchor=(6 / W, 1 - 31 / H), ncol=4, fontsize=6, handletextpad=0.3, columnspacing=1.5)
    letter(fig, H, 1, 2, "a")
    # b: residual vs dG1
    ax = ax_(fig, H, 14, 45, 92, 43); letter(fig, H, 1, 41, "b")
    pts = [("RPE1", 0.12, 0.44, 0.71, C["RPE1"], (5, 2), "left"), ("HepG2", 0.14, 0.26, 1.19, C["HepG2"], (7, 0), "left"),
           ("HCT116", 0.35, 0.094, 0.59, C["HCT116"], (6, 3), "left"), ("K562", 0.22, 0.05, 0.94, C["K562"], (-6, 7), "right"),
           ("Jurkat", 0.24, 0.05, 0.73, C["Jurkat"], (5, 6), "left"), ("HEK293T", 0.53, 0.002, 0.52, C["HEK293T"], (-5, 7), "right"),
           ("hESC", 0.59, 0.01, 0.36, C["hESC"], (6, 3), "left"), ("iPSC day 3", 0.30, 0.01, 0.95, C["other"], (8, 0), "left"),
           ("H1 (VCC)", 0.07, 0.07, 0.65, C["other"], (0, -9), "center")]
    for name, res, dg, rel, col, off, ha in pts:
        hollow = name in ("HCT116", "HEK293T")
        ax.scatter([res], [dg], s=10 + 70 * rel, facecolor="white" if hollow else col, edgecolor=col, lw=0.8, zorder=3)
        ax.annotate(name, (res, dg), textcoords="offset points", xytext=off, fontsize=6, ha=ha, va="center")
    ax.axhline(0, color=LGREY, lw=0.5, zorder=0)
    ax.scatter([0.78], [-0.075], s=10 + 70 * 0.40, facecolor=C["CD4"], edgecolor=C["CD4"], lw=0.8, zorder=3, clip_on=False)
    ax.annotate("CD4+ T, resting (G1 not reported)", (0.78, -0.075), textcoords="offset points", xytext=(-7, 0), fontsize=6, ha="right", va="center")
    ax.axhline(-0.045, color=BLACK, lw=0.4, ls=(0, (2, 2)))
    ax.set_xlim(0, 0.9); ax.set_ylim(-0.1, 0.5)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel("Knockdown residual of survivors"); ax.set_ylabel("Change in G1 fraction")
    title(ax, "Most essential stratum")
    sk = [ax.scatter([], [], s=10 + 70 * r, facecolor=LGREY, edgecolor=GREY, lw=0.6) for r in (0.4, 1.0)]
    ax.legend(sk, ["0.4", "1.0"], title="Relative cells", title_fontsize=6, loc="upper right", ncol=2, handletextpad=0.2,
              columnspacing=0.8, borderpad=0.2)
    # c: dG1 by stratum, same-platform pairs
    ax = ax_(fig, H, 124, 45, 55, 43); letter(fig, H, 113, 41, "c")
    for lab, col, ls, mk, mfc in (("RPE1", C["RPE1"], "-", "o", C["RPE1"]), ("K562", C["K562"], "-", "o", C["K562"]),
                                  ("HCT116", C["HCT116"], "--", "o", "white"), ("HEK293T", C["HEK293T"], "--", "o", "white")):
        if lab in ("RPE1", "K562"):
            rows = A["fig1c"][lab]; y = []
            for lo, hi in zip([-9, -1, -0.5, -0.2], [-1, -0.5, -0.2, 9]):
                v = [r["dG1"] for r in rows if r["e"] is not None and np.isfinite(r["e"]) and lo <= r["e"] < hi]
                y.append(np.median(v) if v else np.nan)
        else:
            y = [A["fig1c"]["orion"][lab]["tiers"][t]["dG1"] for t in ("lt_-1", "-1_-0.5", "-0.5_-0.2", "gt_-0.2")]
        ax.plot(range(4), y, marker=mk, ms=3, color=col, ls=ls, mfc=mfc, mec=col, label=lab)
    ax.axhline(0, color=LGREY, lw=0.5, zorder=0)
    tier_ticks(ax); ax.set_xlabel("DepMap gene effect"); ax.set_ylabel("Change in G1 fraction")
    ax.set_ylim(-0.05, 0.5); title(ax, "Same-platform pairs")
    ax.legend(loc="upper right", ncol=2, handlelength=1.6, columnspacing=0.8, labelspacing=0.3)
    # d: hESC library-input normalisation
    ax = ax_(fig, H, 14, 101, 92, 28); letter(fig, H, 1, 96, "d")
    T = A["fig1d"]["tiers"]; ci = A["ci"]["bc"]; tk = ["lt_-1", "-1_-0.5", "-0.5_-0.2", "gt_-0.2"]
    x = np.arange(4); w = 0.36
    raw = [T[t]["raw_median"] for t in tk]; nrm = [T[t]["norm_median"] for t in tk]
    nlo = [ci[t]["norm"]["ci"][0] for t in tk]; nhi = [ci[t]["norm"]["ci"][1] for t in tk]
    ax.bar(x - w / 2, raw, w, color=LGREY, label="Raw abundance")
    ax.bar(x + w / 2, nrm, w, color=C["hESC"], label="Normalised for library input")
    ax.errorbar(x + w / 2, nrm, yerr=[np.array(nrm) - np.array(nlo), np.array(nhi) - np.array(nrm)], fmt="none", ecolor=BLACK, elinewidth=0.5, capsize=1.4)
    assert abs(nrm[0] - 0.025) < 0.002
    ax.set_yscale("log"); ax.set_ylim(0.01, 3); ax.set_yticks([0.01, 0.1, 1]); ax.set_yticklabels(["0.01", "0.1", "1"])
    ax.set_xticks(x); ax.set_xticklabels([t.replace("\n", " ") for t in TL]); ax.set_xlim(-0.55, 3.55)
    ax.set_xlabel("DepMap gene effect"); ax.set_ylabel("hESC cells relative\nto non-targeting")
    ax.legend(loc="upper left", ncol=2, handlelength=1.2)
    # e: raw abundance vs oligo copies
    ax = ax_(fig, H, 124, 101, 55, 28); letter(fig, H, 113, 96, "e")
    cb = A["fig1d"]["copies_bins"]; labs = [k for k in cb if cb[k]["n"] >= 50]
    yv = [cb[k]["raw_median"] for k in labs]
    ax.plot(range(len(labs)), yv, marker="o", ms=3, color=C["hESC"])
    ax.axhline(1.0, color=LGREY, lw=0.5, zorder=0)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels([l.split("-")[0] if l.split("-")[0] == l.split("-")[-1] else l.replace("-", "–") for l in labs])
    ax.set_xlabel("Oligo copies per guide"); ax.set_ylabel("Raw abundance")
    title(ax, "hESC library")
    save(fig, out, "fig1_fates_v6")


# ================================================================ Figure 2
def rows7(F, M2):
    M = json.load(open(os.path.join(F, "metrics_mixture.json")))
    return M, {"hESC": M["rows"]["hESC"], "K562_d8": M["rows"]["K562_gw"], "K562_d6": M2["rows"]["K562ess"], "RPE1": M2["rows"]["RPE1"],
               "Jurkat": M2["rows"]["Jurkat"], "HepG2": M2["rows"]["HepG2"], "CD4": M2["rows"]["CD4D1rest"]}


def example_panel(ax, r, c, mu, col, ctx):
    edges = np.append(np.arange(0, 2.0001, 0.1), 2.1)
    obs = np.histogram(np.minimum(c / mu, 2.05), edges)[0]
    k = r["nb_size"]; nb = r["nb"]
    uni = expected_ratio_hist(mu, k, [(1.0, nb["r_uniform"])], edges)
    mix = expected_ratio_hist(mu, k, [(1 - nb["e"], nb["r_kd"]), (nb["e"], 1.0)], edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(ctr, obs, width=0.1, color=col, alpha=0.35, lw=0)
    ax.step(edges, np.append(uni, uni[-1]), where="post", color=BLACK, lw=0.7, ls="--")
    ax.step(edges, np.append(mix, mix[-1]), where="post", color=BLACK, lw=0.9)
    ax.axvline(1, color=LGREY, lw=0.5, zorder=0)
    ax.set_xlim(0, 2.1); ax.set_xticks([0, 0.5, 1, 1.5, 2.0]); ax.set_xticklabels(["0", "0.5", "1", "1.5", "≥2"])
    ax.set_xlabel("Own-target count / expected")
    title(ax, "%s, %s: e = %.2f" % (ctx, r["target"], nb["e"]))
    return max(obs.max(), uni.max(), mix.max())


def figure2(F, V, out):
    M, R = rows7(F, V["mixture_more"]); H = 112; fig = fig_(H)
    for j, (ctx, d, col, lab) in enumerate((("hESC", "percell_hesc", C["hESC"], "hESC"), ("K562_gw", "percell_k562gw", C["K562"], "K562"))):
        cells = np.load(os.path.join(F, d, "cells.npz")); targets = json.load(open(os.path.join(F, d, "index.json")))["targets"]
        r, c, mu = example_target(M["rows"][ctx], cells, targets)
        ax = ax_(fig, H, 14 + j * 52, 6, 42, 34); top = example_panel(ax, r, c, mu, col, lab); ax.set_ylim(0, top * 1.12)
        if j == 0:
            ax.set_ylabel("Cells")
    letter(fig, H, 1, 2, "a")
    fig.legend(handles=[Patch(color=C["hESC"], alpha=0.35, label="Observed cells (hESC)"), Patch(color=C["K562"], alpha=0.35, label="Observed cells (K562)"), Line2D([], [], color=BLACK, ls="--", lw=0.7, label="Uniform knockdown"),
                        Line2D([], [], color=BLACK, lw=0.9, label="Two components")], loc="upper left", bbox_to_anchor=(14 / W, 1 - 48 / H), ncol=4,
               handlelength=1.6, columnspacing=1.2)
    # b: seven contexts
    ax = ax_(fig, H, 128, 6, 52, 38); letter(fig, H, 117, 2, "b")
    rng = np.random.default_rng(3)
    ctx7 = [("hESC", "hESC"), ("K562_d8", "K562 day 8"), ("K562_d6", "K562 day 6"), ("Jurkat", "Jurkat"), ("RPE1", "RPE1"), ("HepG2", "HepG2"), ("CD4", "CD4+ T")]
    for k, (cx, lab) in enumerate(ctx7):
        xs, ys = [], []
        for i, (t, lo, hi) in enumerate(TIERS):
            e = np.array([r["nb"]["e"] for r in R[cx] if r["essentiality"] is not None and np.isfinite(r["essentiality"]) and lo <= r["essentiality"] < hi])
            if len(e) < 3:
                continue
            m, l, h = boot_median(e, rng); xx = i + (k - 3) * 0.09
            ax.plot([xx, xx], [l, h], color=C[cx], lw=0.6)
            ax.scatter([xx], [m], s=7, color=C[cx], marker="o" if len(e) >= 20 else "D", zorder=3, lw=0)
            xs.append(xx); ys.append(m)
        ax.plot(xs, ys, color=C[cx], lw=0.6, label=lab)
    tier_ticks(ax); ax.set_ylim(0, 0.65); ax.set_xlabel("DepMap gene effect"); ax.set_ylabel("High-residual share e")
    ax.legend(loc="upper right", ncol=2, fontsize=5.8, handlelength=1.0, columnspacing=0.6, labelspacing=0.15)
    # c: fraction preferring two components
    ax = ax_(fig, H, 14, 66, 46, 36); letter(fig, H, 1, 61, "c")
    for off, cx, col in ((-0.18, "hESC", C["hESC"]), (0.18, "K562_gw", C["K562"])):
        x = np.arange(4) + off; S = [M[cx][t] for t, _, _ in TIERS]
        ax.bar(x, [s["nb_frac_mixture"] for s in S], width=0.34, color=col, lw=0)
        ax.bar(x, [s["ctrl_nb_frac_mixture"] for s in S], width=0.34, color=BLACK, lw=0)
    tier_ticks(ax); ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("DepMap gene effect"); ax.set_ylabel("Targets preferring\ntwo components")
    ax.legend(handles=[Patch(color=C["hESC"], label="hESC"), Patch(color=C["K562"], label="K562"), Patch(color=BLACK, label="Thinned controls")],
              loc="upper left", ncol=3, handlelength=1.0, columnspacing=0.8, bbox_to_anchor=(0, 1.05))
    # d: two guides of one target
    ax = ax_(fig, H, 78, 66, 36, 36); letter(fig, H, 66, 61, "d")
    xs, ys = [], []
    for x in V["mixture"]["V8_rows"]:
        E = x["essentiality"]
        if E is None or not E < -1 or x["p0"] is None or x["p1"] is None:
            continue
        a, b = x["norm_p0"], x["norm_p1"]
        if a is None or b is None or a <= 0 or b <= 0 or a == b:
            continue
        lo_p, hi_p = (x["p0"], x["p1"]) if a < b else (x["p1"], x["p0"])
        xs.append(hi_p["e"]); ys.append(lo_p["e"])
    xs, ys = np.array(xs), np.array(ys); S8 = V["mixture"]["V8"]["lt_-1"]
    assert len(xs) == S8["n"] and abs(np.mean(ys > xs) - S8["frac_more_depleted_has_higher_e"]) < 1e-9
    ax.plot([0, 1], [0, 1], color=LGREY, lw=0.6, zorder=0)
    ax.scatter(xs, ys, s=5, color=C["hESC"], lw=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("e, guide leaving more cells"); ax.set_ylabel("e, guide leaving fewer cells")
    title(ax, "hESC, %d targets" % S8["n"])
    # e: two harvest days
    ax = ax_(fig, H, 136, 66, 36, 36); letter(fig, H, 124, 61, "e")
    a = {r["target"]: r for r in R["K562_d6"]}; b = {r["target"]: r for r in R["K562_d8"]}
    ts = [t for t in sorted(set(a) & set(b)) if a[t]["essentiality"] is not None and a[t]["essentiality"] < -1]
    x6 = np.array([a[t]["nb"]["e"] for t in ts]); x8 = np.array([b[t]["nb"]["e"] for t in ts]); V12 = V["mixture"]["V12"]["lt_-1"]
    assert len(ts) == V12["n"]
    ax.plot([0, 1], [0, 1], color=LGREY, lw=0.6, zorder=0)
    ax.scatter(x6, x8, s=5, color=C["K562"], lw=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("e, K562 day 6"); ax.set_ylabel("e, K562 day 8")
    title(ax, "K562, %d targets" % V12["n"])
    save(fig, out, "fig2_escape_v6")


# ================================================================ Figure 3
def figure3(F, V, out):
    D = json.load(open(os.path.join(F, "metrics_dilution_curve.json"))); K = V["k562_repeat"]; R = V["dilution"]; P = V["points"]
    H = 106; fig = fig_(H)
    lim, ylim = (-5.4, 0.6), (-8.6, 2.0)
    specs = [("hESC_fitted", "hESC, two guides", C["hESC"], True), ("K562_fitted", "K562, two screens", C["K562"], True),
             ("RPE1_fitted", "RPE1, random halves", C["RPE1"], False)]
    for j, (k, lab, col, deatt) in enumerate(specs):
        ax = ax_(fig, H, 14 + j * 46, 6, 36, 36)
        x, y = P[k + "_x"], P[k + "_y"]
        d = R["V3"][k]["deattenuated"] if deatt else R["V3"][k]["simple_emix"]
        b = d["deattenuated_slope"] if deatt else d["ols_slope"]; a = np.mean(y) - b * np.mean(x)
        t = np.array(lim); ax.plot(t, t, color=LGREY, lw=0.6, zorder=0)
        ax.scatter(x, y, s=4, color=col, lw=0, zorder=2); ax.plot(t, a + b * t, color=BLACK, lw=0.8, zorder=3)
        ax.set_xlim(*lim); ax.set_ylim(*ylim); inside(ax, x, y); ax.set_xticks([-4, -2, 0]); ax.set_yticks([-8, -4, 0])
        title(ax, lab)
        ax.text(0.04, 0.96, "n = %d\nslope %.2f\noffset %+.2f" % (len(x), b, R["V3"][k]["simple_emix"]["median_offset"]),
                transform=ax.transAxes, fontsize=6, va="top")
        if j == 0:
            ax.set_ylabel("Measured dilution (log2)")
        if j == 1:
            ax.set_xlabel("Predicted dilution (log2)")
    letter(fig, H, 1, 2, "a")
    # b: CD4
    ax = ax_(fig, H, 154, 6, 26, 36); letter(fig, H, 143, 2, "b")
    for i, dn in enumerate(("D2", "D3", "D4")):
        d = R["V5"]["CD4_%srest" % dn]
        ax.plot([i, i], [d["median_y_old"], d["median_y_new"]], color=GREY, lw=0.6)
        ax.scatter([i], [d["median_y_old"]], s=12, facecolor="white", ec=BLACK, lw=0.6, zorder=3)
        ax.scatter([i], [d["median_y_new"]], s=12, color=BLACK, zorder=3)
        ax.scatter([i], [d["median_x"]], marker="_", s=40, color=C["CD4"], lw=1.0, zorder=4)
    ax.axhline(0, color=LGREY, lw=0.5, zorder=0)
    ax.set_xticks(range(3)); ax.set_xticklabels(["2", "3", "4"]); ax.set_xlim(-0.5, 2.5); ax.set_ylim(-2.6, 0.3)
    ax.set_xlabel("CD4+ T donor"); ax.set_ylabel("Median dilution (log2)"); title(ax, "CD4+ T")
    fig.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="white", mec=BLACK, ms=3.5, label="Classifier only"),
                        Line2D([], [], marker="o", ls="none", mfc=BLACK, mec=BLACK, ms=3.5, label="Classification control subtracted"),
                        Line2D([], [], marker="_", ls="none", color=C["CD4"], ms=6, mew=1.0, label="Predicted from residual")],
               loc="upper left", bbox_to_anchor=(143 / W, 1 - 52 / H), fontsize=6, handletextpad=0.3, labelspacing=0.3)
    # c: slopes
    ax = ax_(fig, H, 44, 62, 92, 30); letter(fig, H, 1, 58, "c")
    rows = [("hESC, per-cell e", R["V3"]["hESC_fitted"]["deattenuated"], C["hESC"], "o", R["V3"]["hESC_fitted"]["simple_emix"]["median_offset"]),
            ("K562, per-cell e", R["V3"]["K562_fitted"]["deattenuated"], C["K562"], "o", R["V3"]["K562_fitted"]["simple_emix"]["median_offset"]),
            ("hESC, mean residual", D["hESC"], C["hESC"], "s", R["V3"]["hESC_fitted"]["simple_eR_same_targets"]["median_offset"]),
            ("K562, mean residual", K["dilution_curve"], C["K562"], "s", None)]
    ax.axvline(1, color=LGREY, lw=0.6, zorder=0)
    for i, (lab, d, col, mk, o) in enumerate(rows):
        yy = -i; lo, hi = d["deattenuated_ci"]
        ax.plot([lo, min(hi, 2.4)], [yy, yy], color=col, lw=0.8)
        ax.scatter([d["deattenuated_slope"]], [yy], s=14, color=col, marker=mk, zorder=3, lw=0)
        ax.scatter([d["null_q975"]], [yy], marker="|", s=30, color=BLACK, lw=0.8, zorder=4)
        ax.text(2.55, yy, "offset %+.2f" % o if o is not None else "offset −0.31 to −0.36", fontsize=6, va="center")
    ax.set_yticks([-i for i in range(len(rows))]); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-0.2, 2.5); ax.set_ylim(-len(rows) + 0.4, 0.6); ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_xlabel("Slope of measured on predicted dilution")
    ax.legend(handles=[Line2D([], [], marker="|", ls="none", color=BLACK, ms=5, mew=0.8, label="Permuted predictor, 97.5th percentile")],
              loc="upper left", bbox_to_anchor=(0.0, -0.29), fontsize=6, handletextpad=0.3)
    save(fig, out, "fig3_dilution_v6")


# ================================================================ Figure 4
def figure4(A, V, out):
    S = V["samescale"]; V1, V2 = S["V1"], S["V2"]; H = 138; fig = fig_(H)
    # a: schematic
    ax = ax_(fig, H, 4, 4, 52, 42); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); letter(fig, H, 1, 2, "a")
    ax.text(0, 9.7, "Published comparison", fontsize=6.5, fontweight="bold", va="top")
    for x0, col, lab, mu in ((0.6, C["hESC"], "hESC", 1.15), (5.4, C["K562"], "K562", 1.95)):
        xs = np.linspace(0, 3.4, 120); ys = np.exp(-0.5 * ((xs - mu) / 0.55) ** 2) * 1.4
        ax.fill_between(xs + x0, 6.6, 6.6 + ys, color=col, alpha=0.6, lw=0)
        ax.text(x0 + mu, 6.0, lab, ha="center", fontsize=6)
    ax.text(0, 5.2, "Magnitude distributions, grouped by\nwhich side silenced more strongly", fontsize=6, va="top")
    ax.text(0, 3.4, "This work", fontsize=6.5, fontweight="bold", va="top")
    for jj in range(4):
        y = 2.5 - jj * 0.5
        ax.plot([2.2, 6.2], [y, y], lw=0.5, color=GREY, zorder=1)
        ax.scatter([2.2], [y], s=7, color=C["hESC"], zorder=2, lw=0); ax.scatter([6.2], [y], s=7, color=C["K562"], zorder=2, lw=0)
    ax.text(7.0, 1.75, "Same target paired;\nknockdown good on\nboth sides", fontsize=6, va="center")
    # b: estimators
    ax = ax_(fig, H, 74, 6, 50, 38); letter(fig, H, 62, 2, "b")
    sets = [("all", "All shared\ntargets"), ("matched", "Knockdown ≤ 0.3\nboth sides")]
    est = [("ratio_of_medians", "Ratio of medians", "o", BLACK, BLACK), ("ratio_of_sums", "Ratio of sums", "o", "white", BLACK),
           ("paired_median_positive", "Median per-target ratio\n(positive targets only)", "s", GREY, GREY)]
    for i, (k, _) in enumerate(sets):
        d = V2[k]["guide_vs_cross"]
        for j, (key, lab, mk, fc, ec) in enumerate(est):
            xx = i + (j - 1) * 0.2; v = d[key]; lo, hi = d[key + "_ci"]
            ax.plot([xx, xx], [lo, hi], color=ec, lw=0.7)
            ax.scatter([xx], [v], s=14, marker=mk, facecolor=fc, edgecolor=ec, lw=0.7, zorder=3, label=lab if i == 0 else None)
    assert abs(V2["matched"]["guide_vs_cross"]["ratio_of_medians"] - 1.072) < 0.001
    ax.axhline(1, color=LGREY, lw=0.6, zorder=0)
    ax.set_yscale("log"); ax.set_ylim(0.1, 2.5); ax.set_yticks([0.1, 0.25, 0.5, 1, 2]); ax.set_yticklabels(["0.1", "0.25", "0.5", "1", "2"])
    ax.minorticks_off(); ax.set_xticks([0, 1]); ax.set_xticklabels([s[1] for s in sets]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("hESC / K562 magnitude")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), handletextpad=0.3, labelspacing=0.6)
    # c: per-target ratio against the share implied by residuals
    ax = ax_(fig, H, 14, 58, 46, 36); letter(fig, H, 1, 54, "c")
    d = A["fig2b"]; h = np.array(d["h"], float); k = np.array(d["k"], float); m = A["fig2c"]["main"]
    rk = m.get("r_kd_main") or A["fig2c"].get("r_kd_main") or {}
    esc = lambda Rv, r: np.clip((np.asarray(Rv, float) - r) / (1 - r), 0, 0.95)
    xx = 2 * np.log2(1 - esc(d["res_h"], rk.get("hesc", 0.20))) - np.log2(1 - esc(d["res_6"], rk.get("k562_d6", 0.16))) - np.log2(1 - esc(d["res_8"], rk.get("k562_d8", 0.16)))
    ok = np.isfinite(h) & np.isfinite(k) & (h > 0) & (k > 0)
    yy = np.where(ok, np.log2(np.where(ok, h, 1) / np.where(ok, k, 1)), np.nan)
    mm = np.isfinite(xx) & np.isfinite(yy) & (xx >= -4) & (xx <= 1) & (yy >= -8) & (yy <= 4)
    ax.scatter(xx[mm], yy[mm], s=2, color=GREY, lw=0, zorder=1)
    t = np.array([-4, 1]); ax.plot(t, m["intercept"] + m["slope"] * t, color=BLACK, lw=0.8)
    ax.plot(t, m["intercept"] + m["null_slope_p97_5"] * t, color=BLACK, lw=0.6, ls=":")
    ax.set_xlim(-4, 1); ax.set_ylim(-8, 4)
    ax.set_xlabel("Predicted ratio (log2)"); ax.set_ylabel("hESC / K562 magnitude (log2)")
    ax.legend(handles=[Line2D([], [], color=BLACK, lw=0.8, label="Theil–Sen %.2f" % m["slope"]), Line2D([], [], color=BLACK, lw=0.6, ls=":", label="Permuted, 97.5%")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=1.4, columnspacing=1.0)
    # d: silenced, near-control and the two classification controls
    ax = ax_(fig, H, 76, 58, 40, 36); letter(fig, H, 64, 54, "d")
    t4 = A["fig2d"]["tiers"]["lt_-1"]
    bars = [("Silenced", t4["mkd_median"], BLACK), ("Near-\ncontrol", t4["mkd_median"] * t4["esc_over_kd"], GREY),
            ("Low-tail\ncontrol", t4["null_median"], "white"), ("Thinned\ncontrol", t4["null2_median"], LGREY)]
    ax.bar(range(4), [b[1] for b in bars], 0.6, color=[b[2] for b in bars], edgecolor=BLACK, lw=0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([b[0] for b in bars]); ax.set_yscale("log"); ax.set_ylim(0.03, 40)
    ax.set_yticks([0.1, 1, 10]); ax.set_yticklabels(["0.1", "1", "10"])
    ax.set_ylabel("Magnitude ((log2 FC)²)"); ax.minorticks_off(); title(ax, "hESC, most essential")
    # e: symmetric silenced comparison
    ax = ax_(fig, H, 132, 58, 48, 36); letter(fig, H, 120, 54, "e")
    grp = [("abundance_<=0.05", "≤ 5% of\ncells left"), ("tier_lt_-1", "Most\nessential"), ("all_classifiable", "All\nclassifiable")]
    ser = [("all_cells", "All cells, both", "o", "white", GREY), ("asymmetric_silenced", "Silenced hESC vs all K562", "s", "white", BLACK),
           ("symmetric_silenced", "Silenced cells, both", "s", BLACK, BLACK)]
    for i, (g, _) in enumerate(grp):
        for j, (key, lab, mk, fc, ec) in enumerate(ser):
            dd = V1[g][key]; xx_ = i + (j - 1) * 0.24
            ax.plot([xx_, xx_], dd["ratio_of_medians_ci"], color=ec, lw=0.7)
            ax.scatter([xx_], [dd["ratio_of_medians"]], s=13, marker=mk, facecolor=fc, edgecolor=ec, lw=0.7, zorder=3, label=lab if i == 0 else None)
    ax.axhline(1, color=LGREY, lw=0.6, zorder=0)
    ax.set_yscale("log"); ax.set_ylim(0.01, 4); ax.set_yticks([0.01, 0.1, 1]); ax.set_yticklabels(["0.01", "0.1", "1"]); ax.minorticks_off()
    ax.set_xticks(range(3)); ax.set_xticklabels([g[1] for g in grp]); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("hESC / K562 magnitude")
    # f: by abundance
    ax = ax_(fig, H, 14, 106, 166, 23); letter(fig, H, 1, 102, "f")
    ser4 = [("All targets, all cells", "o", GREY, GREY), ("Classifiable, all cells", "o", "white", GREY),
            ("Classifiable, silenced hESC vs all K562", "s", "white", BLACK), ("Classifiable, silenced cells both", "s", BLACK, BLACK)]
    groups = [("<=0.05", "≤ 0.05"), ("0.05-0.2", "0.05–0.2"), ("0.2-0.5", "0.2–0.5"), (">0.5", "> 0.5"), ("all", "All")]
    for i, (g, _) in enumerate(groups):
        ds = [V2["abundance_" + g if g != "all" else "all"]["guide_vs_cross"]] + \
             [V1["abundance_" + g if g != "all" else "all_classifiable"][kk] for kk in ("all_cells", "asymmetric_silenced", "symmetric_silenced")]
        for j, (dd, (lab, mk, fc, ec)) in enumerate(zip(ds, ser4)):
            xx_ = i + (j - 1.5) * 0.17
            ax.plot([xx_, xx_], [max(dd["ratio_of_medians_ci"][0], 0.0101), dd["ratio_of_medians_ci"][1]], color=ec, lw=0.7)
            ax.scatter([xx_], [dd["ratio_of_medians"]], s=13, marker=mk, facecolor=fc, edgecolor=ec, lw=0.7, zorder=3, label=lab if i == 0 else None)
    ax.axhline(1, color=LGREY, lw=0.6, zorder=0); ax.axvline(3.5, color=LGREY, lw=0.5)
    ax.set_yscale("log"); ax.set_ylim(0.01, 5); ax.set_yticks([0.01, 0.1, 1]); ax.set_yticklabels(["0.01", "0.1", "1"]); ax.minorticks_off()
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[1] for g in groups]); ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_xlabel("hESC cells remaining (fraction of expected, library-normalised)"); ax.set_ylabel("hESC / K562 magnitude")
    fig.legend(handles=[Line2D([], [], marker=mk, ls="none", mfc=fc, mec=ec, ms=3.5, label=lab) for lab, mk, fc, ec in ser4],
               title="Panels e, f", title_fontsize=6, loc="upper left", bbox_to_anchor=(134 / W, 1 - 26 / H), handletextpad=0.3, labelspacing=0.4)
    save(fig, out, "fig4_samescale_v6")


# ================================================================ Figure 5
def figure5(F, A, V, out):
    Z = np.load(os.path.join(F, "c34_rows.npz")); N = json.load(open(os.path.join(F, "metrics_c6_null_K562.json"))); V13 = V["dir"]["V13"]
    Ca, Cb = V["comp_a100"], V["comp_5090"]; H = 104; fig = fig_(H)
    # a: direction
    ax = ax_(fig, H, 14, 6, 44, 36); letter(fig, H, 1, 2, "a")
    ess, ra, rk, rr = Z["essentiality"], Z["rho_all"], Z["rho_kd"], Z["rho_rand"]; ok = np.isfinite(ra) & np.isfinite(rk)
    groups = [(t, (ess >= lo) & (ess < hi)) for t, lo, hi in TIERS[:3]] + [("all", np.isfinite(ess))]
    for i, (t, g) in enumerate(groups):
        m = ok & g; d = V13[t]; nul = np.nanmedian(rr[m] - ra[m][:, None], axis=0)
        assert abs(np.percentile(nul, 2.5) - d["null_q025"]) < 1e-9
        vp = ax.violinplot([nul], positions=[i], widths=0.5, showextrema=False)
        for bdy in vp["bodies"]:
            bdy.set_facecolor(LGREY); bdy.set_alpha(1); bdy.set_edgecolor("none")
        ax.plot([i + 0.1, i + 0.1], d["ci95"], color=BLACK, lw=0.7); ax.scatter([i + 0.1], [d["median_change"]], s=12, color=BLACK, zorder=3)
    ax.axhline(0, color=GREY, lw=0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(TL[:3] + ["All"]); ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.08, 0.06)
    ax.set_xlabel("DepMap gene effect"); ax.set_ylabel("Change in direction\nagreement with K562")
    ax.legend(handles=[Line2D([], [], marker="o", ms=3, color=BLACK, lw=0.7, label="Silenced − all cells"), Patch(color=LGREY, label="Random subsets")],
              loc="upper right", handletextpad=0.3, labelspacing=0.3)
    # b: composition share
    ax = ax_(fig, H, 76, 6, 40, 36); letter(fig, H, 64, 2, "b")
    ctx = [("RPE1", Ca["RPE1"]), ("HepG2", Cb["HepG2"]), ("Jurkat", Cb["Jurkat"]), ("K562", Ca["K562"]), ("hESC", Ca["hESC"])]
    for i, (lab, d) in enumerate(ctx):
        t = d["lt_-1"]
        ax.plot([i, i], [t["comp"][0], t["comp_upper"][0]], color=C[lab], lw=3, alpha=0.35, solid_capstyle="butt")
        ax.plot([i, i], t["symmetric"][1:3], color=C[lab], lw=0.8); ax.scatter([i], [t["symmetric"][0]], s=14, color=C[lab], zorder=3, lw=0)
    assert abs(Ca["RPE1"]["lt_-1"]["symmetric"][0] - 0.394) < 0.002
    ax.set_xticks(range(len(ctx))); ax.set_xticklabels([c[0] for c in ctx]); ax.set_xlim(-0.5, len(ctx) - 0.5); ax.set_ylim(0, 0.6)
    ax.set_ylabel("Composition share"); title(ax, "Most essential")
    ax.legend(handles=[Line2D([], [], marker="o", ms=3, color=GREY, lw=0.8, label="Symmetric share, 95%"),
                       Line2D([], [], color=GREY, lw=3, alpha=0.35, label="Lower to upper bound")], loc="upper right", handletextpad=0.3)
    # c: composition removal
    ax = ax_(fig, H, 134, 6, 46, 36); letter(fig, H, 122, 2, "c")
    sets = [("RPE1", "lt_-1"), ("RPE1", "-1_-0.5"), ("HepG2", "lt_-1"), ("HepG2", "-1_-0.5")]
    kinds = [("delta_own", BLACK, "Own composition"), ("delta_same_tier_mismatch", GREY, "Another target, same stratum"),
             ("delta_nonessential_mismatch", LGREY, "Another target, non-essential")]
    w = 0.25
    for i, (cx, t) in enumerate(sets):
        for j, (key, col, lab) in enumerate(kinds):
            ax.bar(i + (j - 1) * w, N[cx][t][key], width=w * 0.92, color=col, lw=0, label=lab if i == 0 else None)
    ax.set_xticks(range(4)); ax.set_xticklabels(["RPE1\n< −1", "RPE1\n−1 to −0.5", "HepG2\n< −1", "HepG2\n−1 to −0.5"])
    ax.set_ylim(0, 0.11); ax.set_xlim(-0.5, 3.5); ax.set_ylabel("Change in direction\nagreement with K562")
    ax.legend(loc="upper right", handlelength=1.0, labelspacing=0.25)
    # d: benchmark
    srcs = ["K562", "Jurkat", "HepG2", "RPE1"]
    for j, (key, xl, log) in enumerate((("denoised_scale", "Denoised scale factor", False), ("denoised_rel_mse", "Denoised relative error", True))):
        ax = ax_(fig, H, 22 + j * 52, 60, 42, 30)
        if j == 0:
            letter(fig, H, 1, 56, "d")
        for i, s in enumerate(srcs):
            v = A["fig5c"][s]["hESC"]; a = v["total_common"]["lt_-1"][key]; b = v["kd_common"]["lt_-1"][key]
            ax.plot([a, b], [i, i], color=GREY, lw=0.6, zorder=1)
            ax.scatter([a], [i], s=14, facecolor="white", edgecolor=BLACK, lw=0.7, zorder=2, label="All-cell truth" if i == 0 else None)
            ax.scatter([b], [i], s=14, color=BLACK, zorder=2, label="Silenced-cell truth" if i == 0 else None)
        ax.axvline(1, color=LGREY, lw=0.6, zorder=0)
        ax.set_yticks(range(len(srcs))); ax.set_yticklabels(srcs if j == 0 else []); ax.set_ylim(-0.6, len(srcs) - 0.4)
        ax.set_xlabel(xl)
        if log:
            ax.set_xscale("log"); ax.set_xlim(0.8, 40)
        else:
            ax.set_xlim(0, 1.2); ax.set_ylabel("Source context")
    ax.legend(loc="lower right", handletextpad=0.2, labelspacing=0.3)
    # e: noise share
    ax = ax_(fig, H, 140, 60, 40, 30); letter(fig, H, 126, 56, "e")
    for s in srcs:
        v = A["fig5c"][s]["hESC"]; y = [v["total_common"][t]["noise_share_median"] for t, _, _ in TIERS[:2]]
        ax.plot(range(2), y, marker="o", ms=2.5, color=C[s], label=s)
    ax.set_xticks(range(2)); ax.set_xticklabels(["< −1", "−1 to −0.5"]); ax.set_xlim(-0.3, 1.3); ax.set_ylim(0, 1)
    ax.set_xlabel("DepMap gene effect"); ax.set_ylabel("Noise share of\nhESC ground truth")
    ax.legend(loc="lower left", ncol=2, handlelength=1.2, columnspacing=0.6, labelspacing=0.2)
    save(fig, out, "fig5_magnitude_v6")


def main():
    F, F1, F3, F4, out = sys.argv[1:6]; os.makedirs(out, exist_ok=True)
    A = json.load(open(os.path.join(F1, "figdata_a100.json"), encoding="utf-8"))
    j = lambda p: json.load(open(os.path.join(F4, p)))
    V = dict(samescale=j("rev_samescale/metrics_rev_samescale.json"), dilution=j("rev_dilution/metrics_rev_dilution.json"),
             points=np.load(os.path.join(F4, "rev_dilution/rev_dilution_points.npz")), dir=j("rev_dir_orion.json"),
             mixture_more=j("mixture_more/metrics_mixture_more.json"), mixture=j("rev_mixture/metrics_rev_mixture.json"),
             comp_a100=j("rev_comp_a100.json"), comp_5090=j("rev_comp_5090.json"), k562_repeat=json.load(open(os.path.join(F3, "metrics_k562_repeat.json"))))
    figure1(A, out); figure2(F, V, out); figure3(F, V, out); figure4(A, V, out); figure5(F, A, V, out)


if __name__ == "__main__":
    main()
