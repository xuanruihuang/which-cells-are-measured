"""P4 item 2 (fate paper; expectations O0-O3 pre-stated in notes/fate_paper_plan.md, commit 70338ec): Orion HCT116 and
HEK293T as C1 rows, same definitions as findings (bg). Input: orion_extract.py aggregate (orion_agg):
  counts.npy  rows (line * T + target) * 2 + half, cols n, G1, S, G2M, lib_sum, own_target_raw_sum
  sums64.npy  same rows x genes: sum of log1p CP10K over the group's cells
  ctrl_raw.npy / ctrl_lib.npy  per line: control raw-count sums per gene, control library sum
Per line and target (both halves pooled unless stated):
  cells; residual = own raw / (control rate of the target gene x library); dG1 = G1 share - control G1 share;
  response = random-half cross-signal of (mean log1p of the half - mean log1p of the control's same half) over
             response genes (control >= 0.5 CP10K), target gene zeroed (labelled random halves; includes clone energy)
Tiers: DepMap 24Q4 median gene effect over lines (as bg). Per tier: relative cells (median / non-essential median),
residual median (expressed targets: expected own count >= 1), dG1 median, response median ratio to non-essential.
Usage: fate_orion_c1.py <orion_agg dir> <depmap csv> <out json>. Exploratory."""
import sys, json
import numpy as np

TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf))
MIN_CELLS = 20


def essentiality(path):
    with open(path) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        rows = [np.array([float(x) if x else np.nan for x in line.rstrip("\n").split(",")[1:]]) for line in f]
    return dict(zip(sym, np.nanmedian(np.array(rows), 0)))


def main():
    d, dep, outf = sys.argv[1:4]; ess = essentiality(dep)
    idx = json.load(open(d + "/index.json")); lines, targets, genes = idx["lines"], idx["targets"], idx["genes"]
    T = len(targets); gpos = {g: j for j, g in enumerate(genes)}; ct = targets.index(idx["ctrl"])
    cnt = np.load(d + "/counts.npy"); S = np.load(d + "/sums64.npy", mmap_mode="r")
    craw = np.load(d + "/ctrl_raw.npy"); clib = np.load(d + "/ctrl_lib.npy")
    res = {"rules": __doc__}
    for li, L in enumerate(lines):
        row = lambda t, h: (li * T + t) * 2 + h
        rate = craw[li] / clib[li]; cp = rate * 1e4
        resp = np.where(cp >= 0.5)[0]
        cc = cnt[row(ct, 0)] + cnt[row(ct, 1)]
        g1c = cc[1] / cc[0]
        mc = [np.asarray(S[row(ct, h)])[resp] / cnt[row(ct, h)][0] for h in (0, 1)]
        recs = {}
        for t, name in enumerate(targets):
            if t == ct:
                continue
            c = cnt[row(t, 0)] + cnt[row(t, 1)]
            if c[0] < MIN_CELLS or cnt[row(t, 0)][0] < 5 or cnt[row(t, 1)][0] < 5:
                continue
            e = ess.get(name, np.nan)
            if not np.isfinite(e):
                continue
            j = gpos.get(name)
            exp_own = rate[j] * c[4] if j is not None else 0.0
            v = []
            for h in (0, 1):
                x = np.asarray(S[row(t, h)])[resp] / cnt[row(t, h)][0] - mc[h]
                if j is not None:
                    q = np.searchsorted(resp, j)
                    if q < len(resp) and resp[q] == j:
                        x[q] = 0
                v.append(x)
            recs[name] = dict(e=float(e), cells=float(c[0]), dG1=float(c[1] / c[0] - g1c),
                              resid=float(c[5] / exp_own) if exp_own >= 1 else np.nan, resp=float(v[0] @ v[1]))
        ne = [r for r in recs.values() if r["e"] > -0.2]
        base_c = np.median([r["cells"] for r in ne]); base_r = np.median([r["resp"] for r in ne])
        tiers = {}
        for nm, lo, hi in TIERS:
            rs = [r for r in recs.values() if lo <= r["e"] < hi]
            tiers[nm] = dict(n=len(rs), rel_cells=float(np.median([r["cells"] for r in rs]) / base_c),
                             resid=float(np.nanmedian([r["resid"] for r in rs])), dG1=float(np.median([r["dG1"] for r in rs])),
                             resp_ratio=float(np.median([r["resp"] for r in rs]) / base_r))
        res[L] = dict(control_phase=dict(G1=float(g1c), S=float(cc[2] / cc[0]), G2M=float(cc[3] / cc[0])),
                      n_targets=len(recs), n_response_genes=int(len(resp)), tiers=tiers)
        print(L, json.dumps(res[L]["control_phase"]), "targets", len(recs), flush=True)
        for nm, v in tiers.items():
            print("  ", nm, json.dumps({k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()}), flush=True)
    json.dump(res, open(outf, "w"), indent=1)


if __name__ == "__main__":
    main()
