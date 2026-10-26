"""Fate-layer paper v2: main Figures 1-6, one message per figure, following paper/fate_explainer.md section 4.
Style and helpers as plot_fate_figs_v6.py (183 mm wide, Arial 6-7 pt, 8 pt bold panel letters, black text, Okabe-Ito
colours, one colour per cell type, cell groups by fill). Terms: knockdown, knocked-down cells, escaping cells, escape
fraction e, remaining target expression, response size, cell type. Numbers are read from the saved analysis outputs
(the same files as v6) plus transfer_coef.json (fate_transfer_coef.py); key values are asserted.
Usage: plot_fate_figs_v7.py <figdata> <v1 figdata> <v3 figdata> <v4 figdata> <out>."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from scipy.stats import beta as beta_dist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_fate_figs_v2 import TIERS, example_target
from plot_fate_figs_v6 import (C, BLACK, GREY, LGREY, TL, W, fig_, ax_, letter, title, save, inside, boot_median, tier_ticks,
                               rows7, example_panel)

KD, ESC = "#4d4d4d", "white"   # knocked-down cells (filled), escaping cells (open); the only two fills in Fig. 1a


# ================================================================ Figure 1
# Panel a: one pool of cells carrying the guide, once knockdown has acted (10 knocked down, 2 escaped), grows for days.
# Over the days of growth every dividing cell doubles. Escaping cells divide in every outcome, so each sequenced box holds
# 4 of them (2 doubled); what differs is the knocked-down cells: tolerance 20 (they double too), arrest 10 (they stop
# dividing), depletion 2 (most are depleted).
N0 = (10, 2)
OUTCOMES = (("Depletion", 2, False, "Knocked-down cells depleted", "4 of 6"),
            ("Arrest", 10, True, "Knocked-down cells stop dividing; more in G1", "4 of 14"),
            ("Tolerance", 20, False, "Knocked-down cells keep dividing", "4 of 24"))
R_CELL = 1.25   # mm


def scatter_cells(ax, rng, x0, x1, y0, y1, n_kd, n_esc, ring):
    """place n_kd + n_esc non-overlapping cells in the box (mm coordinates)"""
    pts = []
    while len(pts) < n_kd + n_esc:
        p = rng.uniform([x0 + 2, y0 + 1.8], [x1 - 2, y1 - 1.8])
        if all(np.hypot(*(p - q)) > 2 * R_CELL + (1.3 if ring else 0.5) for q in pts):
            pts.append(p)
    order = rng.permutation(len(pts))
    for i, k in enumerate(order):
        x, y = pts[k]; esc = i < n_esc
        ax.add_patch(Circle((x, y), R_CELL, fc=ESC if esc else KD, ec=KD, lw=0.5))
        if ring and not esc:
            ax.add_patch(Circle((x, y), R_CELL + 0.55, fc="none", ec=BLACK, lw=0.5))


def panel1a(fig, H, rng):
    ax = ax_(fig, H, 6, 3, 160, 46); ax.set_xlim(6, 166); ax.set_ylim(46, 0); ax.axis("off")   # mm; kept inside the page width
    box = lambda x0, x1, y0, y1: ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="white", ec=GREY, lw=0.5))
    # start: one pool
    box(8, 36, 13, 35); scatter_cells(ax, rng, 8, 36, 13, 35, N0[0], N0[1], False)
    ax.text(22, 10.5, "cells with the guide,\nonce knockdown has acted", ha="center", va="bottom", fontsize=6)
    ax.text(22, 36.5, "2 of 12 escaped knockdown", ha="center", va="top", fontsize=5.8)
    ax.text(48, 2.5, "days of growth", ha="center", va="bottom", fontsize=6)
    ax.text(100, 2.5, "cells sequenced", ha="center", va="bottom", fontsize=6)
    for j, (lab, n_kd, ring, msg, share) in enumerate(OUTCOMES):
        yc = 10 + j * 14; y0, y1 = yc - 5.5, yc + 5.5
        ax.annotate("", xy=(59, yc), xytext=(37, 24), arrowprops=dict(arrowstyle="-|>", lw=0.6, color=BLACK, shrinkA=0, shrinkB=0))
        ax.text(61, yc, lab, ha="left", va="center", fontsize=6.5, fontweight="bold")
        box(80, 120, y0, y1); scatter_cells(ax, rng, 80, 120, y0, y1, n_kd, 4, ring)
        ax.text(124, yc - 1.6, msg, ha="left", va="center", fontsize=6)
        ax.text(124, yc + 1.8, "escaping cells: %s sequenced" % share, ha="left", va="center", fontsize=6)


def figure1(A, out):
    H = 99; fig = fig_(H); rng = np.random.default_rng(4)
    panel1a(fig, H, rng)
    letter(fig, H, 1, 2, "a")
    fig.legend(handles=[Line2D([], [], marker="o", ls="none", mfc=KD, mec=KD, ms=4, label="Knocked down"),
                        Line2D([], [], marker="o", ls="none", mfc=ESC, mec=KD, ms=4, label="Escaped knockdown"),
                        Line2D([], [], marker="o", ls="none", mfc=KD, mec=BLACK, mew=1.2, ms=5, label="Stopped dividing")],
               loc="upper left", bbox_to_anchor=(80 / W, 1 - 48.5 / H), ncol=3, fontsize=6, handletextpad=0.3, columnspacing=1.5)
    # b: where 13 cell types sit
    ax = ax_(fig, H, 14, 59, 150, 30); letter(fig, H, 1, 55, "b")
    XA = "#6E6E6E"   # the two X-Atlas/Orion lines: open dark grey, so that no colour of the other cell types is reused
    pts = [("RPE1", 0.12, 0.44, 0.71, C["RPE1"], (5, 2), "left"), ("HepG2", 0.14, 0.26, 1.19, C["HepG2"], (7, 0), "left"),
           ("HCT116", 0.35, 0.094, 0.59, XA, (6, 3), "left"), ("K562", 0.22, 0.05, 0.94, C["K562"], (-7, 0), "right"),
           ("Jurkat", 0.24, 0.05, 0.73, C["Jurkat"], (2, 9), "left"), ("HEK293T", 0.53, 0.002, 0.52, XA, (-6, 7), "right"),
           ("hESC", 0.59, 0.01, 0.36, C["hESC"], (6, 6), "left"), ("iPSC day 3", 0.30, 0.01, 0.95, C["other"], (8, 7), "left"),
           ("H1 (VCC)", 0.07, 0.07, 0.65, C["other"], (0, 9), "center")]
    for name, res, dg, rel, col, off, ha in pts:
        hollow = name in ("HCT116", "HEK293T")
        ax.scatter([res], [dg], s=10 + 70 * rel, facecolor="white" if hollow else col, edgecolor=col, lw=0.8, zorder=3)
        ax.annotate("%s, %.2f" % (name, rel), (res, dg), textcoords="offset points", xytext=off, fontsize=6, ha=ha, va="center")
    ax.axhline(0, color=LGREY, lw=0.5, zorder=0)
    ax.set_xlim(0, 0.9); ax.set_ylim(-0.03, 0.5); ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xticklabels([]); ax.set_ylabel("Change in G1 fraction")
    # cell types without a G1 measurement get their own strip on the same x axis, not a place on the G1 scale
    sx = ax_(fig, H, 14, 91, 150, 4); sx.set_xlim(0, 0.9); sx.set_ylim(-1, 1); sx.set_yticks([]); sx.spines["left"].set_visible(False)
    sx.scatter([0.78], [0], s=10 + 70 * 0.40, facecolor=C["CD4"], edgecolor=C["CD4"], lw=0.8, zorder=3, clip_on=False)
    sx.annotate("CD4+ T, resting, 0.40", (0.78, 0), textcoords="offset points", xytext=(-7, 0), fontsize=6, ha="right", va="center")
    sx.text(-0.01, 0, "G1 not\nreported", ha="right", va="center", fontsize=5.5)
    sx.set_xlabel("Remaining target expression (fraction of control)")
    title(ax, "Most essential genes")
    sk = [ax.scatter([], [], s=10 + 70 * r, facecolor=LGREY, edgecolor=GREY, lw=0.6) for r in (0.25, 1.0)]
    ax.legend(sk, ["0.25", "1.0"], title="Relative cell number (after each name)", title_fontsize=6, loc="upper right", ncol=2, handletextpad=0.2,
              columnspacing=0.8, borderpad=0.2)
    save(fig, out, "fig1_outcomes")


# ================================================================ Figure 2
def figure2(F, V, out):
    M, R = rows7(F, V["mixture_more"]); H = 95; fig = fig_(H)
    for j, (ctx, d, col, lab) in enumerate((("hESC", "percell_hesc", C["hESC"], "hESC"), ("K562_gw", "percell_k562gw", C["K562"], "K562 day 8"))):
        cells = np.load(os.path.join(F, d, "cells.npz")); targets = json.load(open(os.path.join(F, d, "index.json")))["targets"]
        r, c, mu = example_target(M["rows"][ctx], cells, targets)
        ax = ax_(fig, H, 14 + j * 50, 6, 40, 32); top = example_panel(ax, r, c, mu, col, lab); ax.set_ylim(0, top * 1.12)
        if j == 0:
            ax.set_ylabel("Cells")
    letter(fig, H, 1, 2, "a")
    fig.legend(handles=[Patch(color=C["hESC"], alpha=0.35, label="hESC cells"), Patch(color=C["K562"], alpha=0.35, label="K562 cells"),
                        Line2D([], [], color=BLACK, ls="--", lw=0.7, label="Uniform knockdown"), Line2D([], [], color=BLACK, lw=0.9, label="Two groups")],
               loc="upper left", bbox_to_anchor=(14 / W, 1 - 46 / H), ncol=4, handlelength=1.6, columnspacing=1.2)
    # b: escape fraction by essentiality, six cell types (resting CD4+ T: too few fitted genes per stratum; Table S2)
    ax = ax_(fig, H, 126, 6, 54, 36); letter(fig, H, 115, 2, "b")
    rng = np.random.default_rng(3)
    ctx7 = [("hESC", "hESC", 1.0), ("K562_d8", "K562 day 8", 1.0), ("K562_d6", "K562 day 6", 0.4), ("Jurkat", "Jurkat", 0.4),
            ("RPE1", "RPE1", 0.4), ("HepG2", "HepG2", 0.4)]
    for k, (cx, lab, al) in enumerate(ctx7):
        xs, ys = [], []
        for i, (t, lo, hi) in enumerate(TIERS):
            e = np.array([r["nb"]["e"] for r in R[cx] if r["essentiality"] is not None and np.isfinite(r["essentiality"]) and lo <= r["essentiality"] < hi])
            if len(e) < 3:
                continue
            m, l, h = boot_median(e, rng); xx = i + (k - 2.5) * 0.1
            ax.plot([xx, xx], [l, h], color=C[cx], lw=0.6, alpha=al)
            ax.scatter([xx], [m], s=7, color=C[cx], marker="o" if len(e) >= 20 else "D", zorder=3, lw=0, alpha=al)
            xs.append(xx); ys.append(m)
        ax.plot(xs, ys, color=C[cx], lw=0.9 if al == 1 else 0.6, alpha=al, label=lab)
    tier_ticks(ax); ax.set_ylim(0, 0.65); ax.set_xlabel("DepMap gene effect"); ax.set_ylabel("Escape fraction e")
    ax.legend(loc="upper right", ncol=2, fontsize=5.8, handlelength=1.0, columnspacing=0.6, labelspacing=0.15)
    # c: the guide that left fewer cells has the higher e
    ax = ax_(fig, H, 30, 61, 30, 30); letter(fig, H, 16, 55, "c")
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
    xs, ys = np.array(xs), np.array(ys); S8 = V["mixture"]["V8"]["lt_-1"]; n = len(xs); k = int((ys > xs).sum())
    assert n == S8["n"] and abs(k / n - S8["frac_more_depleted_has_higher_e"]) < 1e-9
    lo, hi = beta_dist.ppf(0.025, k, n - k + 1), beta_dist.ppf(0.975, k + 1, n - k)
    # drawn like d: one point per gene, its two guides; above the identity line = the guide leaving fewer cells has the higher e
    ax.plot([0, 1], [0, 1], color=LGREY, lw=0.6, zorder=0); ax.scatter(xs, ys, s=5, color=C["hESC"], lw=0)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("e, guide leaving more cells"); ax.set_ylabel("e, guide leaving fewer cells")
    title(ax, "hESC, %d genes\n%d%% above the line (95%%: %d–%d%%)" % (n, round(100 * k / n), round(100 * lo), round(100 * hi)))
    # d: two harvest days
    ax = ax_(fig, H, 126, 61, 30, 30); letter(fig, H, 112, 55, "d")
    a = {r["target"]: r for r in R["K562_d6"]}; b = {r["target"]: r for r in R["K562_d8"]}
    ts = [t for t in sorted(set(a) & set(b)) if a[t]["essentiality"] is not None and a[t]["essentiality"] < -1]
    x6 = np.array([a[t]["nb"]["e"] for t in ts]); x8 = np.array([b[t]["nb"]["e"] for t in ts]); V12 = V["mixture"]["V12"]["lt_-1"]
    assert len(ts) == V12["n"]
    ax.plot([0, 1], [0, 1], color=LGREY, lw=0.6, zorder=0); ax.scatter(x6, x8, s=5, color=C["K562"], lw=0)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("e, day 6"); ax.set_ylabel("e, day 8")
    title(ax, "K562, %d genes\n%d%% above the line" % (V12["n"], round(100 * np.mean(x8 > x6))))
    save(fig, out, "fig2_escape")


# ================================================================ Figure 3
def figure3(V, out):
    R = V["dilution"]; P = V["points"]; H = 54; fig = fig_(H)
    lim, ylim = (-5.4, 0.6), (-6.4, 3.4)
    specs = [("hESC_fitted", "hESC, two guides", C["hESC"], True, "a"), ("K562_fitted", "K562, two screens", C["K562"], True, "b"),
             ("RPE1_fitted", "RPE1, random halves", C["RPE1"], False, "c")]
    for j, (k, lab, col, deatt, L) in enumerate(specs):
        ax = ax_(fig, H, 14 + j * 56, 6, 42, 40); letter(fig, H, 1 + j * 56, 2, L)
        x, y = P[k + "_x"], P[k + "_y"]
        d = R["V3"][k]["deattenuated"] if deatt else R["V3"][k]["simple_emix"]
        b_ls = R["V3"][k]["simple_emix"]["ols_slope"]; b = d["deattenuated_slope"] if deatt else b_ls
        a = np.mean(y) - b_ls * np.mean(x)          # the black line is the least-squares fit; the de-attenuated slope is an upper value
        t = np.array(lim); ax.plot(t, t, color=LGREY, lw=0.6, zorder=0)
        far = y < ylim[0] + 0.2
        ax.scatter(x[~far], y[~far], s=4, color=col, lw=0, zorder=2); ax.plot(t, a + b_ls * t, color=BLACK, lw=0.8, zorder=3)
        for xf, yf in zip(x[far], y[far]):        # points below the axis: drawn at its edge and labelled with their value
            ax.scatter([xf], [ylim[0] + 0.15], s=10, marker="v", color=col, lw=0, zorder=2, clip_on=False)
            ax.annotate("%.1f" % yf, (xf, ylim[0] + 0.15), textcoords="offset points", xytext=(4, 0), fontsize=5.5, va="center")
        ax.set_xlim(*lim); ax.set_ylim(*ylim); ax.set_xticks([-4, -2, 0]); ax.set_yticks([-6, -4, -2, 0])
        title(ax, lab)
        sl = ("slope %.2f (least squares;\n%.2f de-attenuated)" % (b_ls, b)) if deatt else ("slope %.2f (least squares)" % b_ls)
        ax.text(0.04, 0.96, "n = %d\n%s\noffset %+.2f" % (len(x), sl, R["V3"][k]["simple_emix"]["median_offset"]),
                transform=ax.transAxes, fontsize=6, va="top")
        ax.set_xlabel("Dilution predicted from e (log2)")
        if j == 0:
            ax.set_ylabel("Measured dilution (log2)")
    save(fig, out, "fig3_dilution")


# ================================================================ Figure 4
def figure4(V, out):
    S = V["samescale"]; V1, V2 = S["V1"], S["V2"]; H = 52; fig = fig_(H)
    # a: the comparison
    ax = ax_(fig, H, 4, 4, 56, 44); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off"); letter(fig, H, 1, 2, "a")
    ax.text(0, 9.8, "Each gene paired between hESC and K562", fontsize=6.5, va="top")
    bins = ["> 0.5", "0.2–0.5", "0.05–0.2", "≤ 0.05"]
    for jj, (b, nh) in enumerate(zip(bins, (9, 6, 3, 1))):
        y = 7.1 - jj * 1.45
        ax.text(0, y, b, fontsize=6, va="center")
        for i in range(nh):
            ax.scatter([2.4 + i * 0.28], [y], s=5, color=C["hESC"], lw=0, zorder=2)
        for i in range(9):
            ax.scatter([6.4 + i * 0.28], [y], s=5, color=C["K562"], lw=0, zorder=2)
    ax.text(2.4, 9.0, "hESC cells", fontsize=6, va="center"); ax.text(6.4, 9.0, "K562 cells", fontsize=6, va="center")
    ax.text(0, 8.6, "hESC cells\nremaining", fontsize=5.5, va="top")
    ax.text(0, 1.2, "Response size between independent replicates;\nall cells, or knocked-down cells on both sides", fontsize=6, va="center")
    # b: all shared genes vs knockdown good on both sides
    ax = ax_(fig, H, 76, 6, 34, 36); letter(fig, H, 66, 2, "b")
    for i, k in enumerate(("all", "matched")):
        d = V2[k]["guide_vs_cross"]; v = d["ratio_of_medians"]; lo, hi = d["ratio_of_medians_ci"]
        ax.plot([i, i], [lo, hi], color=BLACK, lw=0.7); ax.scatter([i], [v], s=16, color=BLACK, zorder=3)
    assert abs(V2["matched"]["guide_vs_cross"]["ratio_of_medians"] - 1.072) < 0.001
    ax.axhline(1, color=LGREY, lw=0.6, zorder=0)
    ax.set_yscale("log"); ax.set_ylim(0.01, 5); ax.set_yticks([0.01, 0.1, 1]); ax.set_yticklabels(["0.01", "0.1", "1"])
    ax.minorticks_off(); ax.set_xticks([0, 1]); ax.set_xticklabels(["All shared\ngenes", "Knocked down well\nin both"]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("hESC / K562 response size")
    # c: by hESC cells remaining, all cells vs knocked-down cells on both sides
    ax = ax_(fig, H, 126, 6, 54, 36); letter(fig, H, 116, 2, "c")
    groups = [("<=0.05", "≤ 0.05"), ("0.05-0.2", "0.05–0.2"), ("0.2-0.5", "0.2–0.5"), (">0.5", "> 0.5"), ("all", "All")]
    ser = [("all_cells", "All cells, both sides", "o", "white", GREY), ("symmetric_silenced", "Knocked-down cells, both sides", "o", BLACK, BLACK)]
    for i, (g, _) in enumerate(groups):
        for j, (key, lab, mk, fc, ec) in enumerate(ser):
            dd = V1["abundance_" + g if g != "all" else "all_classifiable"][key]; xx = i + (j - 0.5) * 0.3
            ax.plot([xx, xx], [max(dd["ratio_of_medians_ci"][0], 0.0101), dd["ratio_of_medians_ci"][1]], color=ec, lw=0.7)
            ax.scatter([xx], [dd["ratio_of_medians"]], s=14, marker=mk, facecolor=fc, edgecolor=ec, lw=0.7, zorder=3, label=lab if i == 0 else None)
    assert abs(V1["abundance_<=0.05"]["all_cells"]["ratio_of_medians"] - 0.049) < 0.002 and abs(V1["abundance_<=0.05"]["symmetric_silenced"]["ratio_of_medians"] - 0.22) < 0.01
    ax.axhline(1, color=LGREY, lw=0.6, zorder=0); ax.axvline(3.5, color=LGREY, lw=0.5)
    ax.set_yscale("log"); ax.set_ylim(0.01, 5); ax.set_yticks([0.01, 0.1, 1]); ax.set_yticklabels(["0.01", "0.1", "1"]); ax.minorticks_off()
    ax.set_xticks(range(5)); ax.set_xticklabels([g[1] for g in groups], fontsize=5.8); ax.set_xlim(-0.5, 4.5)
    ax.set_xlabel("hESC cells remaining (fraction of expected)"); ax.set_ylabel("hESC / K562 response size")
    ax.legend(loc="lower right", handletextpad=0.3, labelspacing=0.3, fontsize=5.8)
    save(fig, out, "fig4_hesc")


# ================================================================ Figure 5
def figure5(A, V, out):
    T = V["transfer"]; srcs = ["K562", "Jurkat", "HepG2", "RPE1"]; H = 58; fig = fig_(H)
    # a: single-coefficient transfer model
    ax = ax_(fig, H, 22, 12, 44, 36); letter(fig, H, 1, 2, "a")
    for i, s in enumerate(srcs):
        d = T[s]["lt_-1"]
        for key, fc, lab, yy in (("beta_total", "white", "All cells", i - 0.14), ("beta_kd", BLACK, "Knocked-down cells", i + 0.14)):
            lo, hi = d[key + "_ci95"]
            ax.plot([lo, hi], [yy, yy], color=BLACK, lw=0.6, zorder=1)
            ax.scatter([d[key]], [yy], s=14, facecolor=fc, edgecolor=BLACK, lw=0.7, zorder=3, label=lab if i == 0 else None)
        ax.text(0.66, i, "×%.2f" % d["ratio_kd_over_total"], fontsize=6, va="center")
    assert abs(T["K562"]["lt_-1"]["ratio_kd_over_total"] - 1.65) < 0.01   # guide-half knocked-down truth (2026-10-26)
    ax.set_yticks(range(4)); ax.set_yticklabels(srcs); ax.set_ylim(-0.6, 3.6); ax.set_xlim(0, 0.75); ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("Learned coefficient, source → hESC"); ax.set_ylabel("Source cell type")
    fig.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="white", mec=BLACK, ms=4, label="All cells"),
                        Line2D([], [], marker="o", ls="none", mfc=BLACK, mec=BLACK, ms=4, label="Knocked-down cells")],
               title="hESC ground truth from", title_fontsize=6, loc="upper left", bbox_to_anchor=(22 / W, 1 - 1 / H), ncol=2,
               handletextpad=0.2, columnspacing=1.2, fontsize=5.8)
    # b, c: benchmark
    for j, (key, xl, log, L) in enumerate((("denoised_scale", "Scale factor requested", False, "b"), ("denoised_rel_mse", "Relative error", True, "c"))):
        ax = ax_(fig, H, 88 + j * 52, 12, 38, 36); letter(fig, H, 78 + j * 52, 8, L)
        for i, s in enumerate(srcs):
            v = V["bench"][s]["hESC"]; a = v["total_common"]["lt_-1"][key]; b = v["kd_common"]["lt_-1"][key]
            ci = V["bench_ci"][s]
            for g, yy, val, fc in (("total", i - 0.14, a, "white"), ("kd", i + 0.14, b, BLACK)):   # each ground truth on its own sub-row
                lo, hi = ci[g][key]["ci"]; ax.plot([lo, hi], [yy, yy], color=BLACK, lw=0.6, zorder=1)
                ax.scatter([val], [yy], s=14, facecolor=fc, edgecolor=BLACK, lw=0.7, zorder=2)
        ax.axvline(1, color=LGREY, lw=0.6, zorder=0)
        ax.set_yticks(range(4)); ax.set_yticklabels([]); ax.set_ylim(-0.6, 3.6); ax.set_xlabel(xl)
        if log:
            ax.set_xscale("log"); ax.set_xlim(0.8, 40); ax.set_xticks([1, 10]); ax.set_xticklabels(["1", "10"]); ax.minorticks_off()
            ax.text(1.05, 3.45, "predicting\nno change", fontsize=5.5, va="top")
        else:
            ax.set_xlim(0, 1.2)
    save(fig, out, "fig5_models")


# ================================================================ Figure 6
def figure6(F, V, out):
    Z = np.load(os.path.join(F, "c34_rows.npz")); V13 = V["dir"]["V13"]; Ca, Cb = V["comp_a100"], V["comp_5090"]; H = 52; fig = fig_(H)
    # a: direction with knocked-down cells
    ax = ax_(fig, H, 14, 6, 60, 36); letter(fig, H, 1, 2, "a")
    for i, t in enumerate(["lt_-1", "-1_-0.5", "-0.5_-0.2", "all"]):
        d = V13[t]
        ax.add_patch(Rectangle((i - 0.3, d["null_q025"]), 0.6, d["null_q975"] - d["null_q025"], fc=LGREY, ec="none", zorder=1))
        ax.plot([i, i], d["ci95"], color=BLACK, lw=0.7, zorder=3); ax.scatter([i], [d["median_change"]], s=14, color=BLACK, zorder=4)
    assert abs(V13["lt_-1"]["median_change"] + 0.003) < 0.001
    ax.axhline(0, color=GREY, lw=0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(TL[:3] + ["All"]); ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.08, 0.1); ax.set_yticks([-0.08, -0.04, 0, 0.04, 0.08])
    ax.set_xlabel("DepMap gene effect"); ax.set_ylabel("Change in agreement of\ndirection with K562")
    ax.legend(handles=[Line2D([], [], marker="o", ms=3, color=BLACK, lw=0.7, label="Knocked-down − all cells"), Patch(color=LGREY, label="Random subsets of the same size")],
              loc="upper left", ncol=2, handletextpad=0.3, columnspacing=1.0, fontsize=5.8)
    title(ax, "hESC against K562")
    # b: composition share
    ax = ax_(fig, H, 100, 6, 50, 36); letter(fig, H, 88, 2, "b")
    ctx = [("RPE1", Ca["RPE1"]), ("HepG2", Cb["HepG2"]), ("Jurkat", Cb["Jurkat"]), ("K562", Ca["K562"]), ("hESC", Ca["hESC"])]
    for i, (lab, d) in enumerate(ctx):
        t = d["lt_-1"]
        ax.plot([i, i], [t["comp"][0], t["comp_upper"][0]], color=C[lab], lw=3, alpha=0.35, solid_capstyle="butt")
        ax.plot([i, i], t["symmetric"][1:3], color=C[lab], lw=0.8); ax.scatter([i], [t["symmetric"][0]], s=14, color=C[lab], zorder=3, lw=0)
    assert abs(Ca["RPE1"]["lt_-1"]["symmetric"][0] - 0.394) < 0.002
    ax.set_xticks(range(len(ctx))); ax.set_xticklabels([c[0] for c in ctx]); ax.set_xlim(-0.5, len(ctx) - 0.5); ax.set_ylim(0, 0.6)
    ax.set_ylabel("Share of response from\ncell-cycle composition"); title(ax, "Most essential genes")
    ax.legend(handles=[Line2D([], [], marker="o", ms=3, color=C["RPE1"], lw=0.8, label="Midpoint of the bounds, 95% interval"),
                       Line2D([], [], color=C["RPE1"], lw=3, alpha=0.35, label="Lower to upper bound")], loc="upper right", handletextpad=0.3, fontsize=5.8)
    save(fig, out, "fig6_direction")


def main():
    F, F1, F3, F4, out = sys.argv[1:6]; os.makedirs(out, exist_ok=True)
    A = json.load(open(os.path.join(F1, "figdata_a100.json"), encoding="utf-8"))
    j = lambda p: json.load(open(os.path.join(F4, p)))
    V = dict(samescale=j("rev_samescale/metrics_rev_samescale.json"), dilution=j("rev_dilution/metrics_rev_dilution.json"),
             points=np.load(os.path.join(F4, "rev_dilution/rev_dilution_points.npz")), dir=j("rev_dir_orion.json"),
             mixture_more=j("mixture_more/metrics_mixture_more.json"), mixture=j("rev_mixture/metrics_rev_mixture.json"),
             comp_a100=j("rev_comp_a100.json"), comp_5090=j("rev_comp_5090.json"), transfer=j("transfer_coef.json"),
             # benchmark with the hESC knocked-down truth on guide halves (2026-10-26; run_v2_guidekd.sh)
             bench={s: j("bench_guidekd/w5_benchmark_denoised_%s.json" % s) for s in ("K562", "Jurkat", "HepG2", "RPE1")},
             bench_ci=j("bench_guidekd/ci_bench.json"))
    figure1(A, out); figure2(F, V, out); figure3(V, out); figure4(V, out); figure5(A, V, out); figure6(F, V, out)


if __name__ == "__main__":
    main()
