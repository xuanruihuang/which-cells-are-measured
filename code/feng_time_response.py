"""Is the response present before elimination and lost after it? (A100; fate paper C7 step 4; premise P0 and
expectations F1-F4 pre-registered in notes/fate_paper_plan.md, commit e160a7c.)
Feng et al. 2025 genome-wide fitness-gene CRISPRi screen in iPSC: the same pools sampled on days 3, 4 and 5, the only
early time course in pluripotent cells. Findings (o): no depletion at day 3, elimination under way by day 5.
Cells per target are too few (about 2-20) for per-target vectors, so targets are POOLED within a DepMap tier: one
pseudobulk per (day group, tier, random half). The effect therefore reads as the shared response to knocking down
essential genes, not target-specific wiring.
  reference   cells of the non-essential tier (DepMap > -0.2) of the SAME day group and the SAME half (not all cells of
              the day, which would make the tier effects sum to zero - the template artefact of findings an)
  genes       one COMMON set for every day group: reference cells pooled over all groups >= MIN_CP10K (per-group gene
              sets would make magnitudes incomparable across days)
  effect      log2 of (tier raw sums / tier library) over (reference raw sums / reference library), pseudocount 0.5
  magnitude   cross-signal of the two halves (the halves share no cells and no reference cells)
  residual    tier own-target raw sums / expected, expected from the rate of each target gene in all other cells of the
              same day group (as feng_time.py; control cells are only 12-24 and unusable)
Day groups: P78_D3 / P78_D4 / P78_D5 (same pools, true time course) and P3_D3 vs P4_D5 (same lines, separate pools).
P0 (premise): cell bootstrap of the magnitude, NBOOT draws, 95% interval; the earlier-vs-later difference must have
non-overlapping intervals. F4 (cell-number control): the earlier day downsampled to the later day's cell count, NDOWN
draws, median.
Usage: feng_time_response.py <counts_dir> <agg_with_csc> <out_dir> [screen]. Exploratory."""
import os, sys, json, gzip, re, zlib
import numpy as np
from strata import essentiality_of as essentiality

TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf))
GROUPS = {"P78_D3": [("7", "3"), ("8", "3")], "P78_D4": [("7", "4"), ("8", "4")], "P78_D5": [("7", "5"), ("8", "5")],
          "P3_D3": [("3", "3")], "P4_D5": [("4", "5")]}
PAIRS = (("P78_D3", "P78_D5"), ("P78_D3", "P78_D4"), ("P3_D3", "P4_D5"))
MIN_CP10K = 0.5
NBOOT, NDOWN = 200, 20
EPS = 0.5


def eff(Rt, Lt, Rc, Lc):
    return np.log2((Rt + EPS) / Lt) - np.log2((Rc + EPS) / Lc)


def magnitude(X, lib, h, tier_sel, ref_sel):
    """Cross-half signal of the pooled-tier effect against the same-half reference cells."""
    E = []
    for u in (0, 1):
        a = tier_sel & (h == u); b = ref_sel & (h == u)
        if a.sum() < 5 or b.sum() < 5:
            return np.nan
        E.append(eff(X[a].sum(0), lib[a].sum(), X[b].sum(0), lib[b].sum()))
    return float(E[0] @ E[1])


def main():
    cin, agg, out = sys.argv[1:4]
    screen = sys.argv[4] if len(sys.argv) > 4 else "GenomeWideScreen_FitnessGenes"
    os.makedirs(out, exist_ok=True); cdir = agg + "/csc"
    nm = json.load(open(cdir + "/names.json")); cells, rows = nm["cells"], nm["genes"]
    ptr = np.load(cdir + "/indptr.npy"); idx = np.load(cdir + "/idx.npy", mmap_mode="r"); val = np.load(cdir + "/val.npy", mmap_mode="r")
    nc = len(cells); sym = [r.split(":")[1] for r in rows]; G = len(rows)
    meta = {}
    with gzip.open("%s/%s_Cell-Metadata.tsv.gz" % (cin, screen), "rt") as h:
        next(h)
        for line in h:
            cid, batch, call, cl = line.rstrip("\n").split("\t")
            m = re.search(r"_Pool(\d+)_Day(\d+)", batch)
            meta[re.sub(r"^MP-", "", cid)] = (call, m.group(1), m.group(2))
    gname = list(GROUPS); gof = {pd: k for k, g in enumerate(gname) for pd in GROUPS[g]}
    targets = sorted({v[0].rsplit("_", 1)[0] for v in meta.values() if "_" in v[0]})
    tpos = {t: k for k, t in enumerate(targets)}
    t_of = np.full(nc, -1); d_of = np.full(nc, -1)
    for k, c in enumerate(cells):
        v = meta.get(c)
        if v and "_" in v[0] and (v[1], v[2]) in gof:
            t_of[k] = tpos[v[0].rsplit("_", 1)[0]]; d_of[k] = gof[(v[1], v[2])]
    half = np.array([zlib.crc32(c.encode()) & 1 for c in cells])
    lib = np.zeros(nc)
    for j in range(G):
        s = slice(ptr[j], ptr[j + 1]); lib += np.bincount(idx[s], weights=val[s], minlength=nc)
    ess = essentiality(set(targets)); e = np.array([ess.get(t, np.nan) for t in targets])
    tier_of = np.full(len(targets), -1)
    for q, (nmq, lo, hi) in enumerate(TIERS):
        tier_of[np.isfinite(e) & (e >= lo) & (e < hi)] = q
    ok = (t_of >= 0) & (lib > 0) & (tier_of[np.maximum(t_of, 0)] >= 0)
    ct = np.where(ok, tier_of[np.maximum(t_of, 0)], -1)
    print("assigned cells with a tier", int(ok.sum()), "targets with DepMap", int((tier_of >= 0).sum()), flush=True)
    # common gene set from the reference (non-essential) cells of all groups, then a dense submatrix of the kept cells
    refall = ok & (ct == len(TIERS) - 1)
    rate = np.zeros(G); rl = lib[refall].sum()
    for j in range(G):
        s = slice(ptr[j], ptr[j + 1]); ii = np.asarray(idx[s]); v = np.asarray(val[s], np.float64)
        m = refall[ii]
        if m.any():
            rate[j] = v[m].sum()
    sel = np.where(rate / max(rl, 1) * 1e4 >= MIN_CP10K)[0]
    keep = np.where(ok)[0]; pos = np.full(nc, -1); pos[keep] = np.arange(len(keep))
    X = np.zeros((len(keep), len(sel)), np.float32)
    for q, j in enumerate(sel):
        s = slice(ptr[j], ptr[j + 1]); ii = np.asarray(idx[s]); v = np.asarray(val[s], np.float32)
        m = pos[ii] >= 0
        if m.any():
            X[pos[ii[m]], q] = v[m]
    libk = lib[keep]; hk = half[keep]; dk = d_of[keep]; ctk = ct[keep]; tk = t_of[keep]
    print("dense matrix", X.shape, "genes kept", len(sel), flush=True)
    # own-target residual per (day group, tier), using all kept cells as the reference pool
    gp = {}
    for j, s_ in enumerate(sym):
        gp.setdefault(s_, j)
    own = np.zeros((len(gname), len(TIERS))); expd = np.zeros((len(gname), len(TIERS)))
    for t, name in enumerate(targets):
        if name not in gp or tier_of[t] < 0:
            continue
        q = tier_of[t]; j = gp[name]; s = slice(ptr[j], ptr[j + 1])
        ii = np.asarray(idx[s]); v = np.asarray(val[s], np.float64); m = pos[ii] >= 0
        ii, v = pos[ii[m]], v[m]
        mine = tk[ii] == t
        own[:, q] += np.bincount(dk[ii[mine]], weights=v[mine], minlength=len(gname))
        ref = np.bincount(dk[ii[~mine]], weights=v[~mine], minlength=len(gname))
        libt = np.zeros(len(gname)); np.add.at(libt, dk[tk == t], libk[tk == t])
        tot = np.zeros(len(gname)); np.add.at(tot, dk, libk)
        expd[:, q] += libt * (ref / np.maximum(tot - libt, 1))
    res = {"rules": __doc__, "n_genes": int(len(sel)), "groups": {}, "pairs": {}}
    rq = len(TIERS) - 1
    for k, g in enumerate(gname):
        gm = dk == k
        if gm.sum() == 0:
            continue
        row = {"n_cells": int(gm.sum()), "tiers": {}}
        for q, (nmq, lo, hi) in enumerate(TIERS):
            ts = gm & (ctk == q)
            row["tiers"][nmq] = dict(n=int(ts.sum()), share=float(ts.sum() / gm.sum()),
                                     residual=float(own[k, q] / expd[k, q]) if expd[k, q] > 0 else None,
                                     magnitude=magnitude(X, libk, hk, ts, gm & (ctk == rq)) if q != rq else None)
        res["groups"][g] = row
        print(g, json.dumps({a: {x: (round(y, 4) if isinstance(y, float) else y) for x, y in b.items()} for a, b in row["tiers"].items()}), flush=True)
    # P0 bootstrap and F4 downsampling for each pair and tier
    rng = np.random.default_rng(3)
    for ga, gb in PAIRS:
        ka, kb = gname.index(ga), gname.index(gb)
        if (dk == ka).sum() == 0 or (dk == kb).sum() == 0:
            continue
        pr = {}
        for q, (nmq, lo, hi) in enumerate(TIERS[:-1]):
            out_q = {}
            for lab, kk in ((ga, ka), (gb, kb)):
                ts = (dk == kk) & (ctk == q); rf = (dk == kk) & (ctk == rq)
                bs = []
                ti, ri = np.where(ts)[0], np.where(rf)[0]
                for _ in range(NBOOT):
                    sb = np.zeros(len(keep), bool); sb[rng.choice(ti, len(ti), replace=True)] = True
                    rb = np.zeros(len(keep), bool); rb[rng.choice(ri, len(ri), replace=True)] = True
                    bs.append(magnitude(X, libk, hk, sb, rb))
                bs = np.array([b for b in bs if np.isfinite(b)])
                out_q[lab] = dict(n=int(ts.sum()), magnitude=magnitude(X, libk, hk, ts, rf),
                                  ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) > 10 else None)
            # F4: downsample the earlier group to the later group's cell count
            ta, tb = (dk == ka) & (ctk == q), (dk == kb) & (ctk == q)
            if tb.sum() and ta.sum() > tb.sum():
                ia = np.where(ta)[0]; ds = []
                for _ in range(NDOWN):
                    sb = np.zeros(len(keep), bool); sb[rng.choice(ia, int(tb.sum()), replace=False)] = True
                    ds.append(magnitude(X, libk, hk, sb, (dk == ka) & (ctk == rq)))
                ds = np.array([d for d in ds if np.isfinite(d)])
                out_q["downsampled_earlier"] = float(np.median(ds)) if len(ds) else None
            pr[nmq] = out_q
        res["pairs"]["%s_vs_%s" % (ga, gb)] = pr
        print("PAIR", ga, "vs", gb, flush=True)
        for nmq, v in pr.items():
            print("  ", nmq, json.dumps({a: ({x: (round(y, 3) if isinstance(y, float) else ([round(z, 3) for z in y] if isinstance(y, list) else y)) for x, y in b.items()} if isinstance(b, dict) else round(b, 3) if isinstance(b, float) else b) for a, b in v.items()}), flush=True)
    json.dump(res, open(out + "/metrics_feng_time_response.json", "w"), indent=1)


if __name__ == "__main__":
    main()
