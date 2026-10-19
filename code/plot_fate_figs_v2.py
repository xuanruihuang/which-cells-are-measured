"""New panels of the fate-layer paper v2 (2026-10-19): Fig. 3 (mixture vs uniform partial knockdown), Fig. 5d (library
share of heavy distortion) and Fig. 6a-b (residual-to-dilution curve; direction with silenced cells and composition
removal). Legends in paper/fate_manuscript_v2.md; numbers in notes/fate_analysis_redesign.md §5.1-5.4 and findings (br).
Every number is read from the saved analysis outputs, copied unchanged into <figdata dir>:
  metrics_mixture.json, percell_hesc/{cells.npz,index.json}, percell_k562gw/{cells.npz,index.json}
                                                      (A100 fate_iu/mixture, fate_iu/percell_*; §5.3)
  metrics_libshare.json                               (5090 cross_context/libshare; §5.1)
  metrics_dilution_curve.json, dilution_curve_points.npz   (A100 fate_iu/dilution_curve; §5.2)
  metrics_c34_agree.json, c34_rows.npz                (A100 fate_iu/c34; §5.4)
  metrics_c6_null_K562.json                           (A100 fate_iu/xctx; findings br)
Example targets of Fig. 3a are chosen by a fixed rule, not by eye: in the most essential stratum of each context, the
target with the most cells among those with mean residual 0.45-0.55 (hESC LSM7, K562 RPS12). Fits shown are the saved
negative binomial fits (shared per-gene size k from control cells).
Style as plot_fate_figs.py (notes/fate_paper_figures.md §2).
Usage: plot_fate_figs_v2.py <figdata dir> <out dir> [<v1 figdata dir> <supplement out dir>]; the optional pair also
writes the assembled fig5_budget_v2 and fig6_diagnosis_v2 and Supplementary figS4_mechanism.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gammaln
from matplotlib.patches import Patch

RED, BLUE, GREY = "#c0392b", "#2c3e50", "#7f8c8d"
ARREST, TOL, ELIM = "#2980b9", "#95a5a6", "#8e44ad"
TOL_D = "#5d6d6e"                     # darker edge for the tolerate grey, so K562 reads against white
MM = 1 / 25.4
TIERS = [("lt_-1", -9, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, 9)]
TIERLAB = ["< -1", "-1 to\n-0.5", "-0.5 to\n-0.2", "> -0.2"]
LIBTIERS = ["< -1", "-1 to -0.5", "-0.5 to -0.2", "> -0.2"]

plt.rcParams.update({"font.size": 7, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "axes.linewidth": 0.6, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2,
                     "ytick.major.size": 2.2, "legend.frameon": False, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})


def panel(ax, letter, dx=-0.2, dy=1.05):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", name, flush=True)


def lpmf(c, m, k):
    """Negative binomial log pmf with mean m and size k (as in code/a100/fate_mixture.py)."""
    m = np.maximum(m, 1e-12)
    return gammaln(c + k) - gammaln(k) - gammaln(c + 1) + k * np.log(k / (k + m)) + c * np.log(m / (k + m))


def tier_of(e):
    for lab, lo, hi in TIERS:
        if lo <= e < hi:
            return lab
    return None


# ---------------------------------------------------------------- Figure 3
def example_target(rows, cells, targets):
    """Most essential stratum, mean residual 0.45-0.55, most cells."""
    cand = [r for r in rows if np.isfinite(r["essentiality"]) and r["essentiality"] < -1 and 0.45 <= r["residual"] <= 0.55]
    r = max(cand, key=lambda r: r["n"])
    j = targets.index(r["target"]); m = cells["tc"] == j
    c, mu = cells["own"][m].astype(float), cells["mu"][m]
    assert len(c) == r["n"] and abs(c.sum() / mu.sum() - r["residual"]) < 1e-6
    return r, c, mu


def expected_ratio_hist(mu, k, comps, edges):
    """Expected number of cells per bin of count / mu under a mixture [(weight, rate), ...], summed exactly over cells."""
    out = np.zeros(len(edges) - 1)
    for u in mu:
        cmax = int(np.ceil(edges[-1] * u)) + 1
        cc = np.arange(cmax + 1, dtype=float)
        p = sum(w * np.exp(lpmf(cc, rate * u, k)) for w, rate in comps)
        p[-1] += max(0.0, 1 - p.sum())                       # tail beyond the last edge goes to the overflow bin
        b = np.clip(np.searchsorted(edges, cc / u, side="right") - 1, 0, len(edges) - 2)
        np.add.at(out, b, p)
    return out


def fig3_example(ax, r, c, mu, col, ctx):
    edges = np.append(np.arange(0, 2.0001, 0.1), 2.1)        # last bin collects >= 2
    obs = np.histogram(np.minimum(c / mu, 2.05), edges)[0]
    k = r["nb_size"]; nb = r["nb"]
    uni = expected_ratio_hist(mu, k, [(1.0, nb["r_uniform"])], edges)
    mix = expected_ratio_hist(mu, k, [(1 - nb["e"], nb["r_kd"]), (nb["e"], 1.0)], edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(ctr, obs, width=0.1, color=col, alpha=0.35, lw=0, label="observed cells")
    ax.step(edges, np.append(uni, uni[-1]), where="post", color="black", lw=0.8, ls="--", label="uniform knockdown")
    ax.step(edges, np.append(mix, mix[-1]), where="post", color=col, lw=1.0, label="silenced + unsilenced")
    ax.axvline(1, color=GREY, lw=0.5, ls=":")
    ax.set_xlim(0, 2.1); ax.set_xticks([0, 0.5, 1, 1.5, 2.0])
    ax.set_xticklabels(["0", "0.5", "1\ncontrol", "1.5", ">2"])
    ax.set_xlabel("own-target count / expected")
    ax.set_title("%s, %s" % (r["target"], ctx), loc="left", fontsize=7.5)
    ax.text(0.98, 0.5 if ctx == "hESC" else 0.97, "mean residual %.2f\nunsilenced share %.2f\nΔBIC %.0f" % (r["residual"], nb["e"], nb["dbic"]),
            transform=ax.transAxes, ha="right", va="top", fontsize=6, color="black")


def figure3(F, out):
    M = json.load(open(os.path.join(F, "metrics_mixture.json")))
    fig = plt.figure(figsize=(183 * MM, 58 * MM))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.05, 1.35], wspace=0.48, left=0.055, right=0.99, top=0.86,
                          bottom=0.2)
    for j, (ctx, d, col, lab) in enumerate((("hESC", "percell_hesc", ELIM, "hESC"),
                                             ("K562_gw", "percell_k562gw", TOL_D, "K562 day 8"))):
        cells = np.load(os.path.join(F, d, "cells.npz")); targets = json.load(open(os.path.join(F, d, "index.json")))["targets"]
        r, c, mu = example_target(M["rows"][ctx], cells, targets)
        ax = fig.add_subplot(gs[0, j]); clean(ax); fig3_example(ax, r, c, mu, col, lab)
        if j == 0:
            panel(ax, "a", dx=-0.3); ax.set_ylabel("cells")
            ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.02), handlelength=1.5, labelspacing=0.2, fontsize=5.8)

    # b: escape fraction (negative binomial mixture) per target by stratum
    ax = fig.add_subplot(gs[0, 2]); clean(ax); panel(ax, "b", dx=-0.28)
    for off, ctx, col, ec, lab in ((-0.18, "hESC", ELIM, ELIM, "hESC"), (0.18, "K562_gw", TOL, TOL_D, "K562 day 8")):
        rows = M["rows"][ctx]
        for i, (t, lo, hi) in enumerate(TIERS):
            e = np.array([r["nb"]["e"] for r in rows if np.isfinite(r["essentiality"]) and lo <= r["essentiality"] < hi])
            med = np.median(e); assert abs(med - M[ctx][t]["median_e_nb"]) < 1e-9
            bp = ax.boxplot([e], positions=[i + off], widths=0.3, showfliers=False, patch_artist=True,
                            medianprops=dict(color="black", lw=0.9), whiskerprops=dict(lw=0.5, color=ec),
                            capprops=dict(lw=0.5, color=ec), boxprops=dict(lw=0.5, color=ec))
            bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_alpha(0.55)
        ax.plot([], [], "s", color=col, alpha=0.7, ms=5, label=lab)
    ax.set_xticks(range(4)); ax.set_xticklabels(TIERLAB, fontsize=6.3); ax.set_xlim(-0.6, 3.6)
    ax.set_xlabel("DepMap gene effect"); ax.set_ylabel("unsilenced share e"); ax.set_ylim(0, 1)
    ax.legend(loc="upper right", handletextpad=0.2, labelspacing=0.2)
    ax.set_title("unsilenced share", loc="left", fontsize=7.5)

    # c: fraction of targets preferring the mixture, NB and Poisson, against thinned controls
    ax = fig.add_subplot(gs[0, 3]); clean(ax); panel(ax, "c", dx=-0.2)
    for off, ctx, col, ec in ((-0.17, "hESC", ELIM, ELIM), (0.17, "K562_gw", TOL, TOL_D)):
        x = np.arange(4) + off; S = [M[ctx][t] for t, _, _ in TIERS]
        ax.bar(x, [s["nb_frac_mixture"] for s in S], width=0.32, color=col, alpha=0.8, ec=ec, lw=0.4)
        ax.scatter(x, [s["poisson_frac_mixture"] for s in S], marker="_", s=40, color="black", lw=1.0, zorder=3)
        ax.bar(x, [s["ctrl_nb_frac_mixture"] for s in S], width=0.32, color=GREY, lw=0, zorder=3)
        ax.scatter(x, [s["ctrl_poisson_frac_mixture"] for s in S], marker="_", s=40, color=GREY, lw=1.0, zorder=3)
    hh = [Patch(color=ELIM, alpha=0.8, label="hESC, negative binomial"),
          Patch(fc=TOL, alpha=0.8, ec=TOL_D, lw=0.4, label="K562 day 8, negative binomial"),
          plt.Line2D([], [], marker="_", ms=7, mew=1.0, ls="none", color="black", label="Poisson version"),
          Patch(color=GREY, label="thinned control cells (bar NB, tick Poisson)")]
    ax.set_xticks(range(4)); ax.set_xticklabels(TIERLAB, fontsize=6.3); ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 1.32); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("DepMap gene effect"); ax.set_ylabel("fraction of targets preferring\nsilenced + unsilenced (ΔBIC > 10)")
    ax.legend(handles=hh, loc="upper left", ncol=2, fontsize=5.7, handlelength=1.2, columnspacing=0.8, labelspacing=0.2,
              bbox_to_anchor=(-0.02, 1.06))
    save(fig, out, "fig3_mixture")


# ---------------------------------------------------------------- Figure 5d
def figure5d(F, out):
    L = json.load(open(os.path.join(F, "metrics_libshare.json")))
    fig, axs = plt.subplots(1, 3, figsize=(89 * MM, 52 * MM), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.8, bottom=0.3, wspace=0.12)
    for j, (ctx, lab, col) in enumerate((("hESC", "hESC\n(essential-enriched)", ELIM),
                                         ("K562_gw_day8", "K562 day 8\n(genome-scale)", TOL_D),
                                         ("CD4_D1_rest", "CD4+ T, donor 1\n(genome-scale)", ELIM))):
        ax = axs[j]; clean(ax); S = L[ctx]["strata"]
        floor = S["> -0.2"]["heavy"]
        i_ = np.array([S[t]["i_unmeasurable"] for t in LIBTIERS]); iii = np.array([S[t]["iii_d_below_half"] for t in LIBTIERS])
        assert np.allclose(i_ + iii, [S[t]["heavy"] for t in LIBTIERS])
        x = np.arange(4)
        ax.bar(x, i_, width=0.68, color=col, lw=0, label="too few cells to measure")
        ax.bar(x, iii, bottom=i_, width=0.68, color=col, alpha=0.38, lw=0, label="predicted dilution (1 - e)² < 0.5")
        ax.axhline(floor, color="black", lw=0.6, ls="--")
        excess = sum(S[t]["share_of_library"] * (S[t]["heavy"] - floor) for t in LIBTIERS)
        for xi, t in zip(x, LIBTIERS):
            ax.text(xi, -0.3, "%.0f%%" % (100 * S[t]["share_of_library"]), ha="center", va="top", fontsize=5.6,
                    color=GREY, transform=ax.get_xaxis_transform())
        ax.set_xticks(x); ax.set_xticklabels(["< -1", "-1 to\n-0.5", "-0.5 to\n-0.2", "> -0.2"], fontsize=5.6)
        ax.set_title(lab, fontsize=6.6, pad=2)
        ax.text(0.98, 0.98, "excess, library-\nweighted: %.0f%%" % (100 * excess) if excess >= 0.095 else
                "excess, library-\nweighted: %.1f%%" % (100 * excess), transform=ax.transAxes, ha="right", va="top",
                fontsize=5.8)
        ax.set_ylim(0, 1.0)
        if j == 0:
            ax.set_ylabel("fraction of targets with\nheavy distortion")
            ax.text(-0.42, -0.3, "share of\nlibrary", transform=ax.transAxes, fontsize=5.6, color=GREY, va="top",
                    ha="left")
    axs[1].set_xlabel("DepMap gene effect", labelpad=17)
    h, l = axs[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="black", lw=0.6, ls="--")); l.append("non-essential level (noise floor)")
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=5.6, bbox_to_anchor=(0.56, 1.0), handlelength=1.4,
               columnspacing=0.8)
    panel(axs[0], "d", dx=-0.62, dy=1.12)
    save(fig, out, "fig5d_libshare")


# ---------------------------------------------------------------- Figure 6a-b
def figure6(F, out):
    D = json.load(open(os.path.join(F, "metrics_dilution_curve.json")))
    P = np.load(os.path.join(F, "dilution_curve_points.npz"))
    M = json.load(open(os.path.join(F, "metrics_mixture.json")))
    C = json.load(open(os.path.join(F, "metrics_c34_agree.json"))); Z = np.load(os.path.join(F, "c34_rows.npz"))
    N = json.load(open(os.path.join(F, "metrics_c6_null_K562.json")))
    fig = plt.figure(figsize=(183 * MM, 112 * MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], height_ratios=[1, 0.9], hspace=0.62, wspace=0.42,
                          left=0.065, right=0.99, top=0.93, bottom=0.1)
    panels6ab(fig, gs, D, P, M, C, Z, N)
    save(fig, out, "fig6_diagnosis")


def panels6ab(fig, gs, D, P, M, C, Z, N):
    """Fig. 6a (row 0 of gs, three columns) and 6b (row 1: column 0, columns 1-2)."""
    # a (i): hESC measured vs predicted dilution
    ax = fig.add_subplot(gs[0, 0]); clean(ax); panel(ax, "a", dx=-0.2)
    h = D["hESC"]; x, y = P["hESC_x"], P["hESC_y"]; assert len(x) == h["n"]
    lim = (-3.8, 1.2)
    t = np.linspace(*lim, 2)
    e0 = (M["K562_gw"]["gt_-0.2"]["median_e_nb"], M["hESC"]["gt_-0.2"]["median_e_nb"])   # 0.09, 0.17 (Fig. 3b)
    off = [2 * np.log2(1 - e) for e in e0]
    ax.fill_between(t, t + off[0], t + off[1], color=GREY, alpha=0.18, lw=0)
    ax.plot(t, t, color=GREY, lw=0.6, ls="--")
    ax.scatter(x, y, s=7, color=ELIM, alpha=0.65, lw=0, zorder=3)
    b = h["deattenuated_slope"]; a = y.mean() - b * x.mean()
    ax.plot(t, a + b * t, color=ELIM, lw=1.0)
    kx, ky = D["K562_point"]["x_median"], D["K562_point"]["y_median"]
    ax.scatter([kx], [ky], marker="D", s=22, color=TOL, ec=TOL_D, lw=0.6, zorder=4)
    ax.annotate("K562, most essential\n(median, few cells lost)", (kx, ky), xytext=(0.0, -2.6), fontsize=5.8,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY))
    ax.text(-3.7, 1.25, "measured = predicted", fontsize=5.8, color=GREY, va="top")
    ax.text(0.98, 0.04, "baseline unsilenced share\ne0 = %.2f-%.2f (Fig. 3b)" % e0, transform=ax.transAxes,
            fontsize=5.8, color=GREY, ha="right")
    ax.text(0.02, 0.86, "hESC, %d targets\nslope %.2f (%.2f-%.2f),\nde-attenuated" % ((h["n"], b) + tuple(h["deattenuated_ci"])),
            transform=ax.transAxes, fontsize=6, color=ELIM, va="top")
    ax.set_xlim(*lim); ax.set_ylim(-5.5, 1.3)
    assert x.min() > lim[0] and y.min() > -5.5 and y.max() < 1.3
    ax.set_xlabel("predicted dilution, log2 (1 - e)²\n(from mean survivor residual)")
    ax.set_ylabel("measured dilution, log2\n(all cells / silenced cells)")

    # a (ii): CD4 donors 2-4
    ax = fig.add_subplot(gs[0, 1]); clean(ax)
    for dn, mk in (("D2", "o"), ("D3", "s"), ("D4", "^")):
        k = "CD4_%srest" % dn
        ax.scatter(P[k + "_x"], P[k + "_y"], s=8, marker=mk, facecolor="none", ec=ELIM, lw=0.5, alpha=0.8,
                   label="donor %s (ρ halves %.2f)" % (dn[1], D[k]["rho_halves"]))
    t2 = np.array([-5, 0.5]); ax.plot(t2, t2, color=GREY, lw=0.6, ls="--")
    ax.set_xlim(-5, 0.5); ax.set_ylim(-8.4, 2.8)
    for dn in ("D2", "D3", "D4"):
        k = "CD4_%srest" % dn
        assert P[k + "_x"].min() > -5 and P[k + "_y"].min() > -8.4 and P[k + "_y"].max() < 2.8
    ax.set_xlabel("predicted dilution, log2 (1 - e)²"); ax.set_ylabel("measured dilution, log2")
    ax.set_title("CD4+ T: residual does not\nreproduce between guides", loc="left", fontsize=7)
    ax.legend(loc="lower left", fontsize=5.6, handletextpad=0.1, labelspacing=0.15, borderaxespad=0.1)

    # a (iii): de-attenuated slopes against the permutation null
    ax = fig.add_subplot(gs[0, 2]); clean(ax)
    rows = [("hESC", "hESC"), ("CD4_D2rest", "CD4 donor 2"), ("CD4_D3rest", "CD4 donor 3"), ("CD4_D4rest", "CD4 donor 4")]
    lo_, hi_ = -3, 4
    ax.axvspan(0.5, 1.5, color=GREY, alpha=0.12, lw=0)
    ax.axvline(1, color=GREY, lw=0.6, ls="--")
    for i, (k, lab) in enumerate(rows):
        d = D[k]; yy = -i; col = ELIM
        lo, hi = d["deattenuated_ci"]
        ax.plot([max(lo, lo_), min(hi, hi_)], [yy, yy], color=col, lw=0.9)
        for v, s in ((lo, -1), (hi, 1)):
            if (s < 0 and v < lo_) or (s > 0 and v > hi_):
                ax.annotate("", xy=((lo_ if s < 0 else hi_), yy), xytext=((lo_ + 0.4 if s < 0 else hi_ - 0.4), yy),
                            arrowprops=dict(arrowstyle="-|>", lw=0.7, color=col, mutation_scale=6))
        s_ = d["deattenuated_slope"]
        ax.scatter([np.clip(s_, lo_, hi_)], [yy], s=16, color=col, zorder=3)
        if not lo_ <= s_ <= hi_:
            ax.text(lo_ + 0.15, yy + 0.18, "%.1f" % s_, fontsize=5.6, color=col)
        ax.scatter([d["null_q975"]], [yy], marker="|", s=60, color="black", lw=1.0, zorder=4)
    ax.scatter([], [], marker="|", s=40, color="black", label="permuted residual, 97.5%")
    ax.set_yticks([-i for i in range(4)]); ax.set_yticklabels([r[1] for r in rows])
    ax.set_xlim(lo_, hi_); ax.set_ylim(-4.3, 0.6)
    ax.set_xlabel("de-attenuated slope (95% interval)")
    ax.text(1.0, 0.45, "model\nslope 1", fontsize=5.6, color=GREY, ha="center", va="bottom")
    ax.legend(loc="lower left", fontsize=5.6, bbox_to_anchor=(0.0, -0.02), handletextpad=0.2)

    # b (i): direction agreement with K562, silenced cells minus all cells, against size-matched random subsets
    ax = fig.add_subplot(gs[1, 0]); clean(ax); panel(ax, "b", dx=-0.2)
    ess, ra, rk, rr, rn = Z["essentiality"], Z["rho_all"], Z["rho_kd"], Z["rho_rand"], Z["rho_kd_null2"]
    ok = np.isfinite(ra) & np.isfinite(rk)
    groups = [(t, (ess >= lo) & (ess < hi)) for t, lo, hi in TIERS[:3]] + [("all", np.ones_like(ok))]
    for i, (t, g) in enumerate(groups):
        m = ok & g; base = np.median(ra[m])
        obs = np.median(rk[m]) - base
        nul = np.array([np.nanmedian(rr[m, j]) for j in range(rr.shape[1])]) - base
        assert m.sum() == C[t]["n"] and abs(obs - C[t]["delta"]) < 1e-9 and abs(np.quantile(nul, 0.975) - C[t]["null_q975"]) < 1e-3
        vp = ax.violinplot([nul], positions=[i], widths=0.55, showextrema=False)
        for bdy in vp["bodies"]:
            bdy.set_facecolor(GREY); bdy.set_alpha(0.35); bdy.set_edgecolor("none")
        ax.plot([i - 0.2, i + 0.2], [np.quantile(nul, 0.975)] * 2, color=GREY, lw=0.8)
        ax.scatter([i], [obs], s=20, color=ELIM, zorder=3)
        ax.scatter([i + 0.22], [C[t]["delta_null2_corrected"]], s=14, facecolor="none", ec=ELIM, lw=0.6, zorder=3)
        ax.text(i, 0.105, "n=%d" % m.sum(), ha="center", fontsize=5.6, color=GREY)
    ax.axhline(0, color="black", lw=0.5)
    hh = [plt.Line2D([], [], marker="o", ms=4, ls="none", color=ELIM, label="silenced cells"),
          plt.Line2D([], [], marker="o", ms=3.5, ls="none", mfc="none", mec=ELIM, mew=0.6,
                     label="silenced, selection control subtracted"),
          Patch(color=GREY, alpha=0.35, label="random subsets of all cells (bar: 97.5%)")]
    ax.set_xticks(range(4)); ax.set_xticklabels(TIERLAB[:3] + ["all"], fontsize=6.3)
    ax.set_xlim(-0.5, 3.6); ax.set_ylim(-0.2, 0.12)
    ax.set_xlabel("DepMap gene effect")
    ax.set_ylabel("change in direction agreement\nwith K562 (Δρ, vs all cells)")
    ax.set_title("hESC: escape does not change direction", loc="left", fontsize=7)
    ax.legend(handles=hh, loc="lower left", fontsize=5.5, labelspacing=0.15, handletextpad=0.2, borderaxespad=0.1)

    # b (ii): composition removal in arresting contexts, own vs another target's composition vector
    ax = fig.add_subplot(gs[1, 1:]); clean(ax)
    sets = [("RPE1", "lt_-1"), ("RPE1", "-1_-0.5"), ("HepG2", "lt_-1"), ("HepG2", "-1_-0.5")]
    kinds = [("delta_own", ARREST, 1.0, "own composition vector"),
             ("delta_same_tier_mismatch", ARREST, 0.4, "another target, same stratum"),
             ("delta_nonessential_mismatch", GREY, 0.8, "another target, non-essential stratum")]
    w = 0.25
    for i, (ctx, t) in enumerate(sets):
        for j, (key, col, al, lab) in enumerate(kinds):
            v = N[ctx][t][key]
            ax.bar(i + (j - 1) * w, v, width=w * 0.92, color=col, alpha=al, lw=0, label=lab if i == 0 else None)
            ax.text(i + (j - 1) * w, v + 0.0015, "%.3f" % v, ha="center", va="bottom", fontsize=5.3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["RPE1\n< -1", "RPE1\n-1 to -0.5", "HepG2\n< -1", "HepG2\n-1 to -0.5"], fontsize=6.3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylim(0, 0.09); ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("change in direction agreement\nwith K562 (Δρ) after removing")
    ax.set_title("arresting contexts: composition removal changes direction, mostly via a shared arrest signature",
                 loc="left", fontsize=7)
    ax.legend(loc="upper right", fontsize=5.8, labelspacing=0.2)


# ---------------------------------------------------------------- assembled v2 Figures 5 and 6, Supplementary Fig. S4
# v1 panels are redrawn from the v1 figure data (figdata_a100.json, figdata_5090.json; code/a100/fate_figdata_a100.py,
# code/fate_figdata_5090.py) with the code of plot_fate_figs.py, copied here so the v1 script and its files stay as
# they are. Only letters, layout and the new panels change; v1 Fig. 3d (cycle genes removed) is not in the v2 legend.
def ci_of(d):
    lo, hi = (d["ci"] if d.get("ci") else (np.nan, np.nan))
    return d["est"], lo, hi


def v1_counterfactual(ax):
    """v1 Fig. 3a."""
    from matplotlib.patches import Rectangle
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
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


def v1_budget_ctx(A, B):
    ci = A["ci"]; c5 = B["ci_5090"]["bm"]
    return [("RPE1", ARREST, ci["bm"]["RPE1"]), ("HepG2", ARREST, c5["HepG2"]), ("Jurkat", TOL, c5["Jurkat"]),
            ("K562", TOL, ci["bm"]["K562"]), ("hESC", ELIM, ci["bm"]["hESC"])]


def v1_comp_share(ax, A, B):
    """v1 Fig. 3b."""
    ctx = v1_budget_ctx(A, B)
    for i, (name, col, d) in enumerate(ctx):
        e, lo, hi = ci_of(d["lt_-1"]["comp_share"])
        ax.errorbar([i], [e], yerr=[[max(e - lo, 0)], [max(hi - e, 0)]], fmt="o", ms=3.4, color=col,
                    elinewidth=0.8, capsize=1.8)
    e, lo, hi = ci_of(A["ci"]["bv"]["lt_-1"]["comp_share"])
    ax.errorbar([3.32], [e], yerr=[[e - lo], [hi - e]], fmt="s", ms=3.2, color=BLUE, elinewidth=0.8, capsize=1.8)
    ax.annotate("K562, two independent screens", xy=(3.32, e), xytext=(1.1, 0.0032), fontsize=5.8, color=BLUE,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=BLUE))
    ax.set_xlim(-0.6, 4.6)
    ax.set_xticks(range(5)); ax.set_xticklabels([c[0] for c in ctx], fontsize=6.3, rotation=30, ha="right")
    ax.set_yscale("log"); ax.set_ylabel("composition share"); ax.set_ylim(0.002, 0.5)
    ax.set_title("composition follows arrest", fontsize=7)


def v1_selection(ax, A, B):
    """v1 Fig. 3c."""
    order = sorted(v1_budget_ctx(A, B), key=lambda c: -c[2]["lt_-1"]["selection"]["est"])
    for i, (name, col, d) in enumerate(order):
        e, lo, hi = ci_of(d["lt_-1"]["selection"])
        ax.errorbar([i], [e], yerr=[[e - lo], [hi - e]], fmt="o", ms=3.4, color=col, elinewidth=0.8, capsize=1.8)
    ax.axhline(1, color="black", lw=0.6, ls="--")
    ax.set_xticks(range(5)); ax.set_xticklabels([c[0] for c in order], fontsize=6.3, rotation=30, ha="right")
    ax.set_ylabel("selection term (silenced / all)")
    ax.set_title("selection follows residual", fontsize=7)


def v1_cycle_removed(ax, A, B):
    """v1 Fig. 3d."""
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


def v1_null_control(ax, A):
    """v1 Fig. 3e."""
    rows = [("hESC", A["fig3"]["hESC|budget_hesc_nc"]["negative_control"]),
            ("CD4 donor 2", A["fig3"]["CD4_D2rest|nc"]["negative_control"]),
            ("CD4 donor 3", A["fig3"]["CD4_D3rest|d34"]["negative_control"]),
            ("CD4 donor 4", A["fig3"]["CD4_D4rest|d34"]["negative_control"])]
    x = np.arange(len(rows)); w = 0.36
    ax.bar(x - w / 2, [r[1]["ratio"] for r in rows], w, color=RED, label="random halves")
    ax.bar(x + w / 2, [r[1]["guide_ratio"] for r in rows], w, color=BLUE, label="guide halves")
    ax.axhline(0.10, color="black", lw=0.7, ls="--")
    ax.text(3.45, 0.115, "pre-stated limit", fontsize=6, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([r[0].replace(" donor ", "\ndonor ") for r in rows], fontsize=6.3)
    ax.set_ylabel("inert targets / non-essential stratum")
    ax.set_ylim(0, 1.45); ax.legend(loc="upper left", ncol=2, columnspacing=0.8)
    ax.set_title("null-perturbation control:\ntwo CD4 donors fail and are excluded", fontsize=7, loc="left")


def panel5d(fig, cell, F, letter="d"):
    """Fig. 5d at the house style (7 pt ticks, >= 6 pt annotations), drawn into one gridspec cell."""
    L = json.load(open(os.path.join(F, "metrics_libshare.json")))
    sub = cell.subgridspec(1, 3, wspace=0.14)
    axs = [fig.add_subplot(sub[0, j]) for j in range(3)]
    for j, (ctx, lab, col) in enumerate((("hESC", "hESC\n(essential-enriched)", ELIM),
                                         ("K562_gw_day8", "K562 day 8\n(genome-scale)", TOL_D),
                                         ("CD4_D1_rest", "CD4+ T, donor 1\n(genome-scale)", ELIM))):
        ax = axs[j]; clean(ax); S = L[ctx]["strata"]
        if j:
            ax.sharey(axs[0]); ax.tick_params(labelleft=False)
        floor = S["> -0.2"]["heavy"]
        i_ = np.array([S[t]["i_unmeasurable"] for t in LIBTIERS]); iii = np.array([S[t]["iii_d_below_half"] for t in LIBTIERS])
        assert np.allclose(i_ + iii, [S[t]["heavy"] for t in LIBTIERS])
        x = np.arange(4)
        ax.bar(x, i_, width=0.7, color=col, lw=0, label="too few cells to measure")
        ax.bar(x, iii, bottom=i_, width=0.7, color=col, alpha=0.38, lw=0, label="predicted dilution (1 - e)² < 0.5")
        ax.axhline(floor, color="black", lw=0.6, ls="--")
        excess = sum(S[t]["share_of_library"] * (S[t]["heavy"] - floor) for t in LIBTIERS)
        for xi, t in zip(x, LIBTIERS):
            ax.annotate("%.0f%%" % (100 * S[t]["share_of_library"]), (xi, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -40), textcoords="offset points", ha="center", va="top", fontsize=6, color=GREY)
        ax.set_xticks(x); ax.set_xticklabels(TIERLAB, fontsize=6.3)
        ax.set_title(lab, fontsize=7, pad=2)
        ax.text(0.98, 0.98, "library-weighted\nexcess %s%%" % (("%.0f" if excess >= 0.095 else "%.1f") % (100 * excess)),
                transform=ax.transAxes, ha="right", va="top", fontsize=6.3)
        ax.set_ylim(0, 1.0)
        if j == 0:
            ax.set_ylabel("fraction of targets\nwith heavy distortion")
            ax.annotate("share of\nlibrary", (0, 0), xycoords="axes fraction", xytext=(-38, -40),
                        textcoords="offset points", fontsize=6, color=GREY, va="top")
            panel(ax, letter, dx=-0.45, dy=1.16)
    axs[1].set_xlabel("DepMap gene effect", labelpad=2)
    h, l = axs[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="black", lw=0.6, ls="--")); l.append("non-essential level (noise floor)")
    axs[1].legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.2), ncol=3, fontsize=6.3, handlelength=1.4,
                  columnspacing=0.9)


def figure5_v2(F, A, B, out):
    """v1 Fig. 3 panels with content unchanged: a counterfactual, b composition share and selection term (v1 b, c),
    c cycle genes removed (v1 d), d null-perturbation control (v1 e); e library share (new, restyled)."""
    fig = plt.figure(figsize=(183 * MM, 172 * MM))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 0.9, 1], hspace=0.62, left=0.075, right=0.99, top=0.96,
                          bottom=0.085)
    top = gs[0].subgridspec(1, 3, wspace=0.45)
    ax = fig.add_subplot(top[0, 0]); v1_counterfactual(ax); panel(ax, "a", dx=-0.1, dy=1.0)
    ax = fig.add_subplot(top[0, 1]); clean(ax); v1_comp_share(ax, A, B); panel(ax, "b", dx=-0.3)
    ax = fig.add_subplot(top[0, 2]); clean(ax); v1_selection(ax, A, B)
    mid = gs[1].subgridspec(1, 2, width_ratios=[1, 2.1], wspace=0.3)
    ax = fig.add_subplot(mid[0, 0]); clean(ax); v1_cycle_removed(ax, A, B); panel(ax, "c", dx=-0.3)
    ax = fig.add_subplot(mid[0, 1]); clean(ax); v1_null_control(ax, A); panel(ax, "d", dx=-0.12, dy=1.16)
    panel5d(fig, gs[2], F, letter="e")
    save(fig, out, "fig5_budget_v2")


def v1_bench_dumbbell(ax, A, key, xlabel, title, log=False):
    """v1 Fig. 5c (denoised scale factor); the same drawing for the denoised relative error."""
    srcs = ["K562", "Jurkat", "HepG2", "RPE1"]
    for i, s in enumerate(srcs):
        v = A["fig5c"][s]["hESC"]
        a = v["total_common"]["lt_-1"][key]; b = v["kd_common"]["lt_-1"][key]
        ax.plot([a, b], [i, i], color=GREY, lw=1.0, zorder=1)
        ax.scatter([a], [i], s=22, color=RED, zorder=2, label="uncorrected truth" if i == 0 else None)
        ax.scatter([b], [i], s=22, color=BLUE, zorder=2, label="silenced-cell truth" if i == 0 else None)
    ax.axvline(1, color="black", lw=0.6, ls="--")
    ax.set_yticks(range(len(srcs))); ax.set_yticklabels(["source: " + s for s in srcs], fontsize=6.5)
    ax.set_xlabel(xlabel); ax.set_ylim(-0.6, len(srcs) - 0.2)
    if log:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=7, loc="left")


def v1_noise_share(ax, A):
    """v1 Fig. 5d."""
    for s, col in zip(["K562", "Jurkat", "HepG2", "RPE1"], (BLUE, TOL, ARREST, ELIM)):
        v = A["fig5c"][s]["hESC"]
        y = [v["total_common"][t]["noise_share_median"] for t, _, _ in TIERS[:2]]
        ax.plot(range(2), y, marker="o", ms=3, lw=1.0, color=col, label=s)
    ax.set_xticks(range(2)); ax.set_xticklabels(TIERLAB[:2], fontsize=6.3); ax.set_xlim(-0.3, 1.3)
    ax.set_xlabel("DepMap stratum (hESC ground truth)")
    ax.set_ylabel("noise share of ‖truth‖²")
    ax.set_ylim(0, 1); ax.legend(loc="lower left", fontsize=6, ncol=2, title="source", title_fontsize=6)
    ax.set_title("depletion leaves the truth mostly noise", fontsize=7, loc="left")


def figure6_v2(F, A, B, out):
    D = json.load(open(os.path.join(F, "metrics_dilution_curve.json")))
    P = np.load(os.path.join(F, "dilution_curve_points.npz"))
    M = json.load(open(os.path.join(F, "metrics_mixture.json")))
    C = json.load(open(os.path.join(F, "metrics_c34_agree.json"))); Z = np.load(os.path.join(F, "c34_rows.npz"))
    N = json.load(open(os.path.join(F, "metrics_c6_null_K562.json")))
    fig = plt.figure(figsize=(183 * MM, 168 * MM))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.25, 1, 1], height_ratios=[1, 0.9, 0.72], hspace=0.68, wspace=0.42,
                          left=0.065, right=0.99, top=0.955, bottom=0.075)
    panels6ab(fig, gs, D, P, M, C, Z, N)
    sub = gs[2, :2].subgridspec(1, 2, wspace=0.08)
    ax = fig.add_subplot(sub[0, 0]); clean(ax); panel(ax, "c", dx=-0.42)
    v1_bench_dumbbell(ax, A, "denoised_scale", "denoised scale factor\nthe benchmark asks for",
                      "the instruction to shrink\npredictions disappears")
    ax.set_xlim(0, 1.25)
    ax.legend(loc="upper center", bbox_to_anchor=(1.0, -0.42), ncol=2, fontsize=6)
    ax2 = fig.add_subplot(sub[0, 1]); clean(ax2)
    v1_bench_dumbbell(ax2, A, "denoised_rel_mse", "denoised relative error\n(1 = predicting zero)",
                      "and the error falls", log=True)
    ax2.set_yticklabels([]); ax2.set_xlim(0.8, 40)
    for s, want in (("K562", (3.13, 1.32)), ("RPE1", (28.8, 7.0)), ("Jurkat", (5.09, 1.51)), ("HepG2", (7.85, 2.72))):
        v = A["fig5c"][s]["hESC"]                       # findings (bs) table
        got = (v["total_common"]["lt_-1"]["denoised_rel_mse"], v["kd_common"]["lt_-1"]["denoised_rel_mse"])
        assert all(abs(g - w_) / w_ < 0.01 for g, w_ in zip(got, want)), (s, got)
    ax = fig.add_subplot(gs[2, 2]); clean(ax); panel(ax, "d", dx=-0.3)
    v1_noise_share(ax, A)
    save(fig, out, "fig6_diagnosis_v2")


def figureS4(A, B, out):
    """v1 Fig. 6 unchanged, saved as Supplementary Fig. S4 (drawn by plot_fate_figs.figure6 itself)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import plot_fate_figs as v1
    v1.save = lambda fig, o, name: save(fig, o, "figS4_mechanism")
    v1.figure6(A, B, out)


def main():
    F, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
    figure3(F, out); figure5d(F, out); figure6(F, out)
    if len(sys.argv) > 4:                                    # <v1 figdata dir> <supplement out dir>
        fd, sup = sys.argv[3:5]; os.makedirs(sup, exist_ok=True)
        A = json.load(open(os.path.join(fd, "figdata_a100.json"), encoding="utf-8"))
        B = json.load(open(os.path.join(fd, "figdata_5090.json"), encoding="utf-8"))
        figure5_v2(F, A, B, out); figure6_v2(F, A, B, out); figureS4(A, B, sup)


if __name__ == "__main__":
    main()
