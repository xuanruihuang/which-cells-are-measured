"""C7 step 3 (fate paper; expectations Y0-Y5 pre-stated in notes/fate_paper_plan.md, commit e431280): within a context,
across targets, does the programme a perturbation induces in the SURVIVORS predict how much it is eliminated beyond
what its dependency predicts?
Inputs: fate_budget.py metrics (per-target cells, residual, essentiality) and fate_vectors_export.py vectors (per-unit
effect vectors over the context's response genes, target gene zeroed).
Per target:
  c      log2(cells / non-essential-tier median cells)
  c_hat  residual of c on the rank of the DepMap gene effect (linear on ranks)
  score  programme score as a CROSS product of the two independent unit sides:
         s = mean_{g in P} E_A,g  x  mean_{g in P} E_B,g, signed by the mean of the two sides, so that sampling noise
         in one side cannot inflate it (noise is independent between sides; a product of two noisy means has zero
         expectation under the null)
  Y0     per-target SE of c_hat by Poisson bootstrap on the cell count (B draws), Cochran Q within each DepMap tier
  Y3     expression-matched random gene sets: each programme gene replaced by a random gene from its control-expression
         decile, same set size, NPERM draws -> interval of the partial Spearman
Partial Spearman of c_hat (or residual r) with the score, controlling the DepMap rank.
Usage: fate_c7_programs.py <metrics json> <vectors dir> <extract dir> <depmap csv> <out json> <label> [...]. Exploratory."""
import sys, json
import numpy as np
from fate_crosscontext import essentiality, load, units, TIERS
from fate_budget import Ctx, control_cp10k, response_genes
from scipy.stats import spearmanr, chi2

NPERM = 200
B = 500
PROGS = dict(
    bh3_apoptosis=["BBC3", "PMAIP1", "BAX", "BAK1", "BID", "BCL2L11", "BCL2L1", "MCL1"],
    p53_targets=["CDKN1A", "MDM2", "GADD45A", "SESN1", "RPS27L", "ZMAT3", "TNFRSF10B", "TP53I3"],
    isr_atf4=["DDIT3", "TRIB3", "CHAC1", "ASNS", "SLC7A11", "PPP1R15A"],
    heat_shock=["HSPA1A", "HSPA1B", "DNAJB1", "HSPH1", "BAG3"],
)


def rank(v):
    v = np.asarray(v, float); o = np.argsort(np.argsort(v)).astype(float); return o


def resid_on(y, z):
    rz = rank(z); return y - np.polyval(np.polyfit(rz, y, 1), rz)


def pspear(x, y, z):
    rx, ry, rz = rank(x), rank(y), rank(z)
    f = lambda v: v - np.polyval(np.polyfit(rz, v, 1), rz)
    return float(spearmanr(f(rx), f(ry))[0])


def main():
    mj, vd, xd, dep, outf = sys.argv[1:6]; M = json.load(open(mj)); ess = essentiality(dep)
    res = {"rules": __doc__}
    rng = np.random.default_rng(7)
    for lab in sys.argv[6:]:
        R = M[lab]["rows_random"]; V = load(vd, lab); genes = V["genes"]; gpos = {g: j for j, g in enumerate(genes)}
        tpos = {t: k for k, t in enumerate(V["targets"])}
        ne = [t for t, r in R.items() if np.isfinite(r["essentiality"]) and r["essentiality"] > -0.2]
        base = float(np.median([R[t]["cells"] for t in ne]))
        # control expression deciles for the matched null, from the vectors' response-gene order
        C = Ctx(xd, lab); sel = response_genes(C); cpall = control_cp10k(C)
        assert [C.genes[q] for q in sel] == genes, "gene order differs from the exported vectors"
        cp = cpall[sel]
        ts, c, e, rr, side = [], [], [], [], []
        for t, r in R.items():
            if t not in tpos or not np.isfinite(r["essentiality"]) or r["cells"] < 20:
                continue
            u = units(V, tpos[t], "total")
            if u is None:
                continue
            ts.append(t); c.append(np.log2(max(r["cells"], 1) / base)); e.append(r["essentiality"])
            rr.append(r["resid"]); side.append(u)
        c = np.array(c); e = np.array(e); rr = np.array(rr); U = np.stack(side)        # targets x 2 x genes
        c_hat = resid_on(c, e)
        print("==", lab, "targets", len(ts), flush=True)
        # Y0: per-target SE of c_hat from a Poisson bootstrap of the cell count, Cochran Q per tier
        cells = np.array([R[t]["cells"] for t in ts], float)
        bsd = np.std([resid_on(np.log2(np.maximum(rng.poisson(cells), 1) / base), e) for _ in range(B)], axis=0)
        y0 = {}
        for nm, lo, hi in TIERS:
            m = (e >= lo) & (e < hi) & (bsd > 0)
            if m.sum() < 10:
                y0[nm] = dict(n=int(m.sum())); continue
            w = 1 / bsd[m] ** 2; q = float((w * (c_hat[m] - (w * c_hat[m]).sum() / w.sum()) ** 2).sum())
            y0[nm] = dict(n=int(m.sum()), Q=q, df=int(m.sum() - 1), p=float(chi2.sf(q, m.sum() - 1)),
                          sd_observed=float(np.std(c_hat[m])), sd_error=float(np.median(bsd[m])))
        def score(cols):
            a = U[:, 0][:, cols].mean(1); b = U[:, 1][:, cols].mean(1)
            return np.sign(a + b) * np.sqrt(np.abs(a * b))
        out = dict(n_targets=len(ts), Y0=y0,
                   Y1_spearman_cells_essentiality=float(spearmanr(c, e)[0]),
                   median_cells=float(np.median(cells)), programmes={})
        for pn, gl in PROGS.items():
            cols = np.array([gpos[g] for g in gl if g in gpos])
            if len(cols) < 3:
                out["programmes"][pn] = dict(n_genes=len(cols), note="too few genes measured"); continue
            s = score(cols)
            obs_c = pspear(c_hat, s, e); obs_r = pspear(rr, s, e)
            null_c, null_r = [], []
            if len(cp) == len(genes):
                dec = np.digitize(np.log10(cp + 1e-3), np.quantile(np.log10(cp + 1e-3), np.linspace(0, 1, 11)[1:-1]))
                for _ in range(NPERM):
                    pick = np.array([rng.choice(np.where(dec == dec[j])[0]) for j in cols])
                    sp = score(pick); null_c.append(pspear(c_hat, sp, e)); null_r.append(pspear(rr, sp, e))
            out["programmes"][pn] = dict(n_genes=int(len(cols)), partial_spearman_chat=obs_c, partial_spearman_resid=obs_r,
                                         null_chat=[float(np.percentile(null_c, 2.5)), float(np.percentile(null_c, 97.5))] if null_c else None,
                                         null_resid=[float(np.percentile(null_r, 2.5)), float(np.percentile(null_r, 97.5))] if null_r else None,
                                         spearman_score_essentiality=float(spearmanr(s, e)[0]))
            print("  %-14s n=%d  chat %.3f null %s | resid %.3f null %s" % (
                pn, len(cols), obs_c, out["programmes"][pn]["null_chat"], obs_r, out["programmes"][pn]["null_resid"]), flush=True)
        print("  Y0", json.dumps(y0), "Y1 %.3f" % out["Y1_spearman_cells_essentiality"], flush=True)
        res[lab] = out
    json.dump(res, open(outf, "w"), indent=1)


if __name__ == "__main__":
    main()
