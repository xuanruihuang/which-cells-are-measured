"""Supplementary-material inputs that the figure export did not carry (2026-10-18): control-cell phase fractions and
proliferation markers per context (Supplementary Figure S1), per-cell classification diagnostics (S2), and the
nine threshold combinations of the same-scale comparison (S2 table). Reshapes saved outputs only, except the
threshold sweep, which recomputes the published E1 / E2 ratios at the stated thresholds from per_target.npz.
Usage: fate_suppdata.py <out json>."""
import json, sys, os
import numpy as np
import config
from fate_budget import Ctx, ALL

F = config.get("FATE_ROOT")
CTX = [(x.split(":")[0], F + "/" + x.split(":")[1]) for x in os.environ.get(
    "FATE_CTX", "K562:budget,RPE1:budget,hESC:budget,K562d8:budget_d8,CD4_D2rest:budget_cd4,"
    "CD4_D3rest:budget_cd4,CD4_D4rest:budget_cd4").split(",")]
PROLIF = ("MKI67", "TOP2A")


def main():
    out = {"rules": __doc__, "phase": {}, "classify": {}}
    for lab, d in CTX:
        try:
            C = Ctx(d, lab)
        except FileNotFoundError:
            print("missing", lab, d, flush=True); continue
        cnt = C.cnt[C.ct]                      # control rows: parity x half x phase x group x [n, lib, own, mu]
        ph = cnt[..., 0].sum(axis=(0, 1, 3))
        A, lib, n = C.sums(C.ct, "random", 0, ALL); B, libb, nb = C.sums(C.ct, "random", 1, ALL)
        cp = (A.sum(0) + B.sum(0)) / (lib.sum() + libb.sum()) * 1e4
        pos = {g: float(cp[C.gq[g]]) for g in PROLIF if g in C.gq}
        grp = C.cnt[:-1][..., 0].sum(axis=(0, 1, 2, 3))       # cells by group over all targets
        out["phase"][lab] = dict(control_cells=int(ph.sum()), G1=float(ph[0] / ph.sum()), S=float(ph[1] / ph.sum()),
                                 G2M=float(ph[2] / ph.sum()), proliferation_cp10k=pos,
                                 rkd=C.idx["rkd"], n_cells=C.idx["n_cells"],
                                 group_cells=dict(knocked_down=float(grp[0]), escaper=float(grp[1]),
                                                  undetermined=float(grp[2])))
        print(lab, json.dumps(out["phase"][lab])[:200], flush=True)

    if os.environ.get("FATE_PHASE_ONLY"):
        json.dump(out, open(sys.argv[1], "w")); print("written (phase only)", sys.argv[1], flush=True); return
    # nine threshold combinations of the same-scale comparison
    z = np.load(F + "/samescale_out/per_target.npz", allow_pickle=True)
    h, k = z["h_guide_lfc"], z["k_cross_lfc"]
    rows = []
    for kd in (0.2, 0.3, 0.4):
        good = (z["res_h"] <= kd) & (z["res_6"] <= kd) & (z["res_8"] <= kd)
        ok = np.isfinite(h) & np.isfinite(k) & (h > 0) & (k > 0)
        m = ok & good
        rows.append(dict(knockdown_threshold=kd, n_matched=int(good.sum()), n_plotted=int(m.sum()),
                         ratio=float(2 ** np.median(np.log2(h[m] / k[m]))) if m.any() else None))
    out["threshold_sweep_knockdown_only"] = rows
    out["note_threshold_sweep"] = ("the published sweep crosses three expression thresholds (0.3 / 1 / 3 CP10K) with "
                                   "three knockdown thresholds; the gene-set axis is fixed in per_target.npz, so only "
                                   "the knockdown axis is recomputed here and the full 3x3 range is quoted from "
                                   "findings (bb): E2 0.89-1.18, E1 0.31-0.36, all targets 0.33-0.41")
    out["classify"] = {k: v for k, v in json.load(open(F + "/kdgroup/metrics_kdgroup.json")).items()
                       if k not in ("rules", "rows")}
    kd_rows = json.load(open(F + "/kdgroup/metrics_kdgroup.json"))["rows"]
    out["classify_rows"] = [dict(target=r["target"], e=r["essentiality"], expected_ctrl=r["expected_ctrl"],
                                 n_kd=r["n_kd"], n_esc=r["n_esc"], n_und=r["n_und"], resid=r["resid"],
                                 m_kd=r.get("m_kd"), m_esc=r.get("m_esc"), m_k562=r.get("m_k562"),
                                 null=r.get("null"), null2=r.get("null2")) for r in kd_rows]
    json.dump(out, open(sys.argv[1], "w"))
    print("written", sys.argv[1], flush=True)


if __name__ == "__main__":
    main()
