"""C7 step 3 follow-up (POST HOC, after reading the pre-registered run; fate_paper_plan.md C7 step 3). Two questions:
(1) the programme sets kept only 3-5 measured genes, so widen them (more synonyms / members) and report a bootstrap CI
    of the partial Spearman next to the expression-matched null;
(2) RPE1's p53-target relation with elimination beyond dependency may just be arrest (p53 -> G1 -> fewer cells), so
    also partial out dG1, and report the correlation of the score with dG1.
Usage: fate_c7_posthoc.py <metrics json> <vectors dir> <extract dir> <depmap csv> <out json> <label> [...]"""
import sys, json
import numpy as np
from fate_crosscontext import essentiality, load, units, TIERS
from fate_budget import Ctx, control_cp10k, response_genes
from scipy.stats import spearmanr

NPERM, NBOOT = 300, 1000
PROGS = dict(
    p53_targets=["CDKN1A", "MDM2", "GADD45A", "SESN1", "SESN2", "RPS27L", "ZMAT3", "TNFRSF10B", "TP53I3", "BTG2",
                 "TP53INP1", "TIGAR", "FAS", "TRIAP1", "CCNG1", "DDB2", "XPC", "POLH", "PHLDA3", "AEN", "EDA2R"],
    bh3_apoptosis=["BBC3", "PMAIP1", "BAX", "BAK1", "BID", "BCL2L11", "BCL2L1", "MCL1", "BCL2", "BIK", "HRK",
                   "CASP3", "CASP7", "CASP9", "APAF1", "DIABLO", "CYCS"],
    isr_atf4=["DDIT3", "TRIB3", "CHAC1", "ASNS", "SLC7A11", "PPP1R15A", "ATF4", "SESN2", "CEBPB", "VEGFA",
              "SLC3A2", "WARS1", "GARS1", "SARS1", "CTH", "MTHFD2", "PSAT1", "PHGDH", "SHMT2"],
    heat_shock=["HSPA1A", "HSPA1B", "DNAJB1", "HSPH1", "BAG3", "HSPB1", "DNAJA1", "HSPA8", "HSP90AA1", "CHORDC1",
                "ZFAND2A", "UBB"],
)


def rank(v):
    return np.argsort(np.argsort(np.asarray(v, float))).astype(float)


def resid_multi(y, Z):
    Z = np.column_stack([np.ones(len(y))] + [rank(z) for z in Z])
    return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def pspear(x, y, Zs):
    return float(spearmanr(resid_multi(rank(x), Zs), resid_multi(rank(y), Zs))[0])


def main():
    mj, vd, xd, dep, outf = sys.argv[1:6]; M = json.load(open(mj)); ess = essentiality(dep)
    rng = np.random.default_rng(11); res = {"rules": __doc__}
    for lab in sys.argv[6:]:
        R = M[lab]["rows_random"]; V = load(vd, lab); genes = V["genes"]; gpos = {g: j for j, g in enumerate(genes)}
        tpos = {t: k for k, t in enumerate(V["targets"])}
        C = Ctx(xd, lab); sel = response_genes(C); cp = control_cp10k(C)[sel]
        ne = [t for t, r in R.items() if np.isfinite(r["essentiality"]) and r["essentiality"] > -0.2]
        base = float(np.median([R[t]["cells"] for t in ne]))
        ts, c, e, dg, U = [], [], [], [], []
        for t, r in R.items():
            if t not in tpos or not np.isfinite(r["essentiality"]) or r["cells"] < 20:
                continue
            u = units(V, tpos[t], "total")
            if u is None:
                continue
            ts.append(t); c.append(np.log2(max(r["cells"], 1) / base)); e.append(r["essentiality"]); dg.append(r["dG1"]); U.append(u)
        c = np.array(c); e = np.array(e); dg = np.array(dg); U = np.stack(U)
        dec = np.digitize(np.log10(cp + 1e-3), np.quantile(np.log10(cp + 1e-3), np.linspace(0, 1, 11)[1:-1]))
        def score(cols):
            a = U[:, 0][:, cols].mean(1); b = U[:, 1][:, cols].mean(1)
            return np.sign(a + b) * np.sqrt(np.abs(a * b))
        out = {"n_targets": len(ts), "programmes": {}}
        for pn, gl in PROGS.items():
            cols = np.array([gpos[g] for g in gl if g in gpos])
            if len(cols) < 5:
                out["programmes"][pn] = dict(n_genes=int(len(cols)), note="fewer than 5 genes measured"); continue
            s = score(cols)
            obs = pspear(c, s, [e]); obs_dg = pspear(c, s, [e, dg])
            bs = [pspear(c[i], s[i], [e[i]]) for i in (rng.integers(0, len(ts), len(ts)) for _ in range(NBOOT))]
            null = [pspear(c, score(np.array([rng.choice(np.where(dec == dec[j])[0]) for j in cols])), [e]) for _ in range(NPERM)]
            out["programmes"][pn] = dict(
                n_genes=int(len(cols)), genes_measured=[genes[j] for j in cols],
                partial_given_ess=obs, boot_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                matched_null=[float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                partial_given_ess_and_dG1=obs_dg, spearman_score_dG1=float(spearmanr(s, dg)[0]),
                spearman_score_ess=float(spearmanr(s, e)[0]))
            p = out["programmes"][pn]
            print("%-6s %-14s n=%2d  partial %.3f CI [%.3f, %.3f] null [%.3f, %.3f] | +dG1 %.3f  rho(score,dG1) %.3f" % (
                lab, pn, len(cols), obs, p["boot_ci"][0], p["boot_ci"][1], p["matched_null"][0], p["matched_null"][1],
                obs_dg, p["spearman_score_dG1"]), flush=True)
        res[lab] = out
    json.dump(res, open(outf, "w"), indent=1)


if __name__ == "__main__":
    main()
