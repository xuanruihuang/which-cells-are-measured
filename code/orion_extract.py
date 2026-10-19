"""X-Atlas/Orion single cells (HCT116, HEK293T; genome-wide dual-guide CRISPRi) -> per (line, target, random half)
aggregates on the A100, for M4 (a clean third pair: same lab, platform and effector; non-essential genes included).
Same conventions as tahoe_extract.py / cd4_sc_extract.py so M4 runs through the cell-line gamma pipeline:
log1p CP10K (library = all counts), own phase calls (Tirosh half 1, seed 7; control sets from the line's PIE control
mean), half = crc32(cell_barcode) mod 2, controls = gene_target 'Non-Targeting'.
Usage:
  orion_extract.py map <data_dir> <out_dir> <workers>    files not yet in <out_dir>/done.txt; one accumulator per worker
  orion_extract.py reduce <out_dir>                         add partials to running totals and delete them
Per group (line, target, half): sum of log1p CP10K over the PIE gene axis (18,151-gene order of the PIE Orion
preprocessed dir), cell count, phase counts, library sum, sum of the raw count of the group's own target gene
(knockdown), phase-resolved sums on p53 targets + Tirosh genes. Per line: control raw-count sums per gene and control
library sum (for the expected own-target count). Target vocabulary: union of gene_target over all files (first pass
on that column only)."""
import os, sys, glob, json, zlib, time
import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from multiprocessing import Pool
from strata import S_GENES, G2M, P53
import config

ROOT = config.get("ORION_ROOT")
PIE = ROOT + "/PIE_x_atlas_orion/preprocessed"
META = ROOT + "/X-Atlas-Orion/metadata/gene_metadata.parquet"
LINES = ["HCT116", "HEK293T"]
CTRL = "Non-Targeting"
G_ = {}


def setup(data_dir):
    meta = json.load(open(PIE + "/meta.json"))
    genes = meta["genes"]; gpos = {g: k for k, g in enumerate(genes)}
    gm = pq.read_table(META).to_pydict()
    tok = np.array([int(t) for t in gm["gene_token_id"]])
    lut = np.full(tok.max() + 1, -1, np.int64)
    for t, s, e in zip(tok, gm["gene_name"], gm["ensembl_id"]):      # PIE gene axis mixes symbols and Ensembl IDs
        if s in gpos:
            lut[t] = gpos[s]
        elif e in gpos:
            lut[t] = gpos[e]
    vocab_file = os.path.join(os.path.dirname(data_dir.rstrip("/")), "target_vocab.json")
    if os.path.exists(vocab_file):
        targets = json.load(open(vocab_file))
    else:
        tg = set()
        for f in sorted(glob.glob(data_dir + "/*.parquet")):
            tg |= set(pq.read_table(f, columns=["gene_target"]).column(0).to_pylist())
        targets = sorted(tg); json.dump(targets, open(vocab_file, "w"))
    tpos = {t: k for k, t in enumerate(targets)}
    ctx = {c: i for i, c in enumerate(LINES)}
    ctrl = np.load(PIE + "/ctrl_means.npy")
    cmap = meta["context_to_id"]            # PIE context name -> row of ctrl_means
    crow = {}
    for c in LINES:
        key = [k for k in cmap if k.lower().startswith(c.lower())]
        crow[c] = cmap[key[0]] if key else 0
    rng = np.random.default_rng(7)
    sS = [g for g in S_GENES if g in gpos]; sS = list(rng.permutation(sS))[: len(sS) // 2]
    sG = [g for g in G2M if g in gpos]; sG = list(rng.permutation(sG))[: len(sG) // 2]
    iS, iG = np.array([gpos[g] for g in sS]), np.array([gpos[g] for g in sG])
    ctl = {}
    for c in LINES:
        mu = ctrl[crow[c]]; bins = np.digitize(mu, np.quantile(mu, np.linspace(0, 1, 26)[1:-1]))
        r = np.random.default_rng(100 + ctx[c])
        ctl[ctx[c]] = tuple(np.concatenate([r.choice(np.where(bins == bins[j])[0], 10, replace=False) for j in ix])
                            for ix in (iS, iG))
    sub = np.array(sorted({gpos[g] for g in P53 + S_GENES + G2M if g in gpos}))
    tgt_col = np.array([gpos.get(t, -1) for t in targets])
    return dict(genes=genes, lut=lut, targets=targets, tpos=tpos, ctx=ctx, iS=iS, iG=iG, ctl=ctl, sub=sub, tgt_col=tgt_col)


def init(data_dir):
    G_.update(setup(data_dir))


def do_chunk(args):
    files, out, tag = args
    C, T, G = len(LINES), len(G_["targets"]), len(G_["genes"])
    acc = np.zeros((C * T * 2, G), np.float32); cnt = np.zeros((C * T * 2, 6))
    sub = G_["sub"]; accp = np.zeros((C * T * 2, 3, len(sub)), np.float32)
    craw = np.zeros((C, G)); clib = np.zeros(C)
    ncell = 0; t0 = time.time()
    for fn in files:
        pf = pq.ParquetFile(fn)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["gene_token_id", "gene_expression", "sample", "gene_target", "cell_barcode"])
            gcol, ecol = t.column("gene_token_id").combine_chunks(), t.column("gene_expression").combine_chunks()
            off = gcol.offsets.to_numpy(); tok = gcol.values.to_numpy(); val = ecol.values.to_numpy().astype(np.float64)
            n = len(off) - 1; ncell += n
            cell = np.repeat(np.arange(n), np.diff(off))
            ok = val > 0
            lib = np.bincount(cell[ok], weights=val[ok], minlength=n)
            gi = np.where(tok < len(G_["lut"]), G_["lut"][np.minimum(tok, len(G_["lut"]) - 1)], -1)
            keep = ok & (gi >= 0)
            Xraw = csr_matrix((val[keep], (cell[keep], gi[keep])), shape=(n, G))
            y = np.log1p(val[keep] / lib[cell[keep]] * 1e4)
            X = csr_matrix((y, (cell[keep], gi[keep])), shape=(n, G))
            smp = t.column("sample").to_pylist()
            line = np.array([G_["ctx"].get(s.split("_")[0], -1) for s in smp])
            tg = np.array([G_["tpos"].get(x, -1) for x in t.column("gene_target").to_pylist()])
            half = np.array([zlib.crc32(b.encode()) & 1 for b in t.column("cell_barcode").to_pylist()])
            ph = np.zeros(n, np.int8)
            sS_all = np.asarray(X[:, G_["iS"]].mean(1)).ravel(); sG_all = np.asarray(X[:, G_["iG"]].mean(1)).ravel()
            for c in np.unique(line[line >= 0]):
                m = np.where(line == c)[0]; cS, cG = G_["ctl"][c]
                a = sS_all[m] - np.asarray(X[m][:, cS].mean(1)).ravel()
                b = sG_all[m] - np.asarray(X[m][:, cG].mean(1)).ravel()
                ph[m] = np.where((a <= 0) & (b <= 0), 0, np.where(a >= b, 1, 2))
            valid = (line >= 0) & (tg >= 0)
            key = (line.astype(np.int64) * T + tg) * 2 + half
            kv = key[valid]; iv = np.where(valid)[0]
            uk, inv = np.unique(kv, return_inverse=True)
            Gm = csr_matrix((np.ones(len(iv)), (inv, iv)), shape=(len(uk), n))
            acc[uk] += (Gm @ X).toarray().astype(np.float32)
            np.add.at(cnt, (uk[inv], 0), 1)
            for q in range(3):
                np.add.at(cnt, (kv[ph[iv] == q], 1 + q), 1)
            np.add.at(cnt, (kv, 4), lib[iv])
            # own-target raw count per cell (knockdown)
            tc = G_["tgt_col"][tg[iv]]
            has = tc >= 0
            own = np.zeros(len(iv))
            if has.any():
                own[has] = np.asarray(Xraw[iv[has], tc[has]]).ravel()
            np.add.at(cnt, (kv, 5), own)
            # control raw sums per line
            for c in np.unique(line[valid]):
                m = iv[(line[iv] == c) & (tg[iv] == G_["tpos"].get(CTRL, -2))]
                if len(m):
                    craw[c] += np.asarray(Xraw[m].sum(0)).ravel(); clib[c] += lib[m].sum()
            Xs = X[:, sub]
            for q in range(3):
                mq = ph[iv] == q
                if mq.any():
                    ukq, invq = np.unique(kv[mq], return_inverse=True)
                    Gq = csr_matrix((np.ones(mq.sum()), (invq, iv[mq])), shape=(len(ukq), n))
                    accp[ukq, q] += (Gq @ Xs).toarray().astype(np.float32)
    np.save(os.path.join(out, "parts", "acc_%s.npy" % tag), acc)
    np.save(os.path.join(out, "parts", "cnt_%s.npy" % tag), cnt)
    np.save(os.path.join(out, "parts", "accp_%s.npy" % tag), accp)
    np.save(os.path.join(out, "parts", "craw_%s.npy" % tag), craw)
    np.save(os.path.join(out, "parts", "clib_%s.npy" % tag), clib)
    return files, ncell, time.time() - t0


def reduce(out, data_dir):
    S = setup(data_dir); C, T, G = len(LINES), len(S["targets"]), len(S["genes"])
    if os.path.exists(out + "/sums64.npy"):
        acc = np.load(out + "/sums64.npy"); cnt = np.load(out + "/counts.npy"); accp = np.load(out + "/phase_sums.npy")
        craw = np.load(out + "/ctrl_raw.npy"); clib = np.load(out + "/ctrl_lib.npy")
    else:
        acc = np.zeros((C * T * 2, G)); cnt = np.zeros((C * T * 2, 6)); accp = np.zeros((C * T * 2, 3, len(S["sub"])))
        craw = np.zeros((C, G)); clib = np.zeros(C)
    parts = sorted(glob.glob(out + "/parts/acc_*.npy"))
    for f in parts:
        acc += np.load(f); cnt += np.load(f.replace("acc_", "cnt_")); accp += np.load(f.replace("acc_", "accp_"))
        craw += np.load(f.replace("acc_", "craw_")); clib += np.load(f.replace("acc_", "clib_"))
    np.save(out + "/sums64.npy.tmp.npy", acc); os.replace(out + "/sums64.npy.tmp.npy", out + "/sums64.npy")
    np.save(out + "/counts.npy", cnt); np.save(out + "/phase_sums.npy", accp)
    np.save(out + "/ctrl_raw.npy", craw); np.save(out + "/ctrl_lib.npy", clib)
    for f in parts:
        for k in ("acc_", "cnt_", "accp_", "craw_", "clib_"):
            os.remove(f.replace("acc_", k))
    json.dump(dict(lines=LINES, targets=S["targets"], genes=S["genes"], ctrl=CTRL,
                   phase_sum_genes=[S["genes"][j] for j in S["sub"]], key="(line * n_targets + target) * 2 + half",
                   counts_cols=["n", "G1", "S", "G2M", "lib_sum", "own_target_raw_sum"]), open(out + "/index.json", "w"))
    print("reduced", len(parts), "parts; cells", int(cnt[:, 0].sum()), flush=True)


def main():
    mode = sys.argv[1]
    if mode == "map":
        data, out, nw = sys.argv[2], sys.argv[3], int(sys.argv[4])
        os.makedirs(out + "/parts", exist_ok=True)
        done = set(open(out + "/done.txt").read().split()) if os.path.exists(out + "/done.txt") else set()
        files = [f for f in sorted(glob.glob(data + "/*.parquet")) if os.path.basename(f) not in done]
        batch = time.strftime("%m%d%H%M%S")
        chunks = [(files[k::nw], out, "%s_%02d" % (batch, k)) for k in range(nw) if files[k::nw]]
        t0 = time.time(); tot = 0
        with Pool(len(chunks), initializer=init, initargs=(data,)) as pool:
            for fs, n, dt in pool.imap_unordered(do_chunk, chunks):
                tot += n
                with open(out + "/done.txt", "a") as fh:
                    fh.write("".join(os.path.basename(f) + "\n" for f in fs))
                print("chunk done", len(fs), "files", n, "cells", "%.0fs" % dt, flush=True)
        print("map done", len(files), "files", tot, "cells", "%.0fs" % (time.time() - t0), flush=True)
    else:
        reduce(sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
