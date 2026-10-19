"""Readers for the Replogle and Nadig h5ad files (obs categories, gene names, raw-count rows).
Lifted unchanged from explore_V_day6_day8.py, which served another analysis in the same working repository."""
import numpy as np
import h5py

CTRL_NAMES = ("non-targeting", "CTRL", "control")
CHUNK = 50000


def dec(v):
    return v.decode() if isinstance(v, bytes) else v


def read_codes_cats(f, key):
    obs = f["obs"]; node = obs[key]
    if isinstance(node, h5py.Group):
        codes = node["codes"][:]; cats = np.array([dec(c) for c in node["categories"][:]])
    else:
        codes = node[:]
        if "__categories" in obs and key in obs["__categories"]:
            cats = np.array([dec(c) for c in obs["__categories"][key][:]])
        else:
            cats, codes = np.unique(np.array([dec(v) for v in codes]), return_inverse=True)
    return codes.astype(np.int64), cats


def var_gene_names(f):
    var = f["var"]
    if "gene_name" in var:
        node = var["gene_name"]
        if isinstance(node, h5py.Group):
            cats, codes = node["categories"][:], node["codes"][:]
            return np.array([dec(cats[c]) for c in codes])
        if "__categories" in var and "gene_name" in var["__categories"]:
            cats, codes = var["__categories"]["gene_name"][:], node[:]
            return np.array([dec(cats[c]) for c in codes])
        return np.array([dec(v) for v in node[:]])
    for k in ("_index", "index", "gene_id"):
        if k in var:
            return np.array([dec(v) for v in var[k][:]])
    raise KeyError("no gene names")


def obs_info(path):
    with h5py.File(path) as f:
        codes, cats = read_codes_cats(f, "gene")
        gc, gcats = read_codes_cats(f, "gem_group") if isinstance(f["obs"]["gem_group"], h5py.Group) or \
            "__categories" in f["obs"] and "gem_group" in f["obs"]["__categories"] else (f["obs"]["gem_group"][:], None)
        genes = var_gene_names(f)
    gem = np.asarray(gc if gcats is None else gcats[gc])
    return codes, list(cats), gem, genes


def read_rows(path, rows, cols):
    """Dense rows x cols (float32) from layers/counts or X (dense or CSR), chunked over sorted rows."""
    with h5py.File(path) as f:
        M = f["layers"]["counts"] if "layers" in f and "counts" in f["layers"] else f["X"]
        out = np.zeros((len(rows), len(cols)), np.float32)
        N = M.shape[0] if isinstance(M, h5py.Dataset) else len(M["indptr"]) - 1
        pos = 0
        for s in range(0, N, CHUNK):
            e = min(N, s + CHUNK)
            sel = rows[(rows >= s) & (rows < e)]
            if not len(sel):
                continue
            if isinstance(M, h5py.Dataset):
                blk = M[s:e][:, cols]
            else:
                ip = M["indptr"][s:e + 1]
                data, ind = M["data"][ip[0]:ip[-1]], M["indices"][ip[0]:ip[-1]]
                full = np.zeros((e - s, M.attrs["shape"][1] if "shape" in M.attrs else int(ind.max()) + 1), np.float32)
                rr = np.repeat(np.arange(e - s), np.diff(ip))
                full[rr, ind] = data
                blk = full[:, cols]
            out[pos:pos + len(sel)] = blk[sel - s]; pos += len(sel)
            print("  rows", e, "/", N, flush=True)
    return out
