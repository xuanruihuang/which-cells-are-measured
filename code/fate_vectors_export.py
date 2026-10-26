"""Export per-target effect vectors from a fate_budget_extract.py layout, for the cross-context consistency test (fate
paper C6 validation; expectations V1-V3 pre-stated in notes/fate_paper_plan.md §3, commit 733bd72). Uses the same
functions as fate_budget.py (response genes, phase-mixture decomposition, thinning null), so the vectors are exactly
those behind the budget.
Per target and unit side (u = 0, 1), over the context's response genes (control >= 0.5 CP10K), target gene zeroed:
  tot_r / comp_r     total effect E and composition effect Ec, random-half units
  kd_r / kdcomp_r    the same from the knocked-down cells only, thinning null subtracted (groupable targets only)
  kd_g               knocked-down effect on guide-parity units (targets with >= 2 guides), the thinning null of random
                     half u subtracted from guide half u (added 2026-10-26 so that guide-unit contexts score both ground
                     truths on the same kind of replicate)
  tot_g / comp_g     total and composition effects, guide-parity units (targets with >= 2 guides)
NaN where a vector is unavailable. Output: <out_dir>/<label>_vectors.npz.
Usage: fate_vectors_export.py <extract dir> <out_dir> <label> [<label> ...]. Exploratory."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fate_budget import Ctx, vectors, null_vector, response_genes, ALL


def main():
    d, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
    for lab in sys.argv[3:]:
        C = Ctx(d, lab)
        sel = response_genes(C); genes = [C.genes[q] for q in sel]; pos = {g: j for j, g in enumerate(genes)}
        tl = C.tl[:-1]; T, G = len(tl), len(sel)
        A = {k: np.full((T, 2, G), np.nan, np.float32) for k in ("tot_r", "comp_r", "kd_r", "kdcomp_r", "tot_g", "comp_g", "kd_g")}
        for k, t in enumerate(tl):
            j = pos.get(t)
            for unit, keys in (("random", ("tot_r", "comp_r")), ("guide", ("tot_g", "comp_g"))):
                if unit == "guide" and C.gpt.get(t, 1) < 2:
                    continue
                V = [vectors(C, k, unit, u, ALL, sel) for u in (0, 1)]
                if any(v is None for v in V):
                    continue
                for u in (0, 1):
                    e, ec = V[u][0].copy(), V[u][1].copy()
                    if j is not None:
                        e[j] = 0; ec[j] = 0
                    A[keys[0]][k, u] = e; A[keys[1]][k, u] = ec
            K = [vectors(C, k, "random", u, [0], sel) for u in (0, 1)]
            nl = [null_vector(C, k, u, sel) for u in (0, 1)]
            if all(v is not None for v in K) and all(v is not None for v in nl):
                for u in (0, 1):
                    e, ec = K[u][0] - nl[u], K[u][1].copy()
                    if j is not None:
                        e[j] = 0; ec[j] = 0
                    A["kd_r"][k, u] = e; A["kdcomp_r"][k, u] = ec
                if C.gpt.get(t, 1) >= 2:
                    Kg = [vectors(C, k, "guide", u, [0], sel) for u in (0, 1)]
                    if all(v is not None for v in Kg):
                        for u in (0, 1):
                            e = Kg[u][0] - nl[u]
                            if j is not None:
                                e[j] = 0
                            A["kd_g"][k, u] = e
        np.savez("%s/%s_vectors.npz" % (out, lab), targets=np.array(tl), genes=np.array(genes), **A)
        print(lab, "targets", T, "genes", G, "with random", int(np.isfinite(A["tot_r"][:, 0, 0]).sum()),
              "guide", int(np.isfinite(A["tot_g"][:, 0, 0]).sum()), "kd", int(np.isfinite(A["kd_r"][:, 0, 0]).sum()), flush=True)


if __name__ == "__main__":
    main()
