"""Check that escape_fraction.py reproduces the paper's mixture fits target by target.

mixture mode: refits every analysable target of the per-cell extractions with escape_fraction.fit_perturbation, in the
order and with the random stream of code/fate_mixture.py (one generator, seed 57, hESC first), and compares e, the BIC
difference and the thinned-control BIC difference with metrics_mixture.json.
purity mode: refits the hESC guide-purity thirds as code/fate_mixture_guideconf.py does (ties broken with a generator
seeded by the target index) and compares e for all cells and per third with metrics_mixture_guideconf.json.
Usage:
  check_escape_fraction.py mixture <percell_hesc dir> <percell_k562gw dir> <metrics_mixture.json>
  check_escape_fraction.py purity <percell_guideconf dir> <percell_hesc dir> <metrics_mixture_guideconf.json>
Exit status 1 if any value differs by more than 1e-6."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "escape_fraction"))
import escape_fraction as EF

TOL = 1e-6


def load(d, lam_from_index=False):
    idx = json.load(open(d + "/index.json")); ctrl = np.load(d + "/ctrl.npz"); cells = np.load(d + "/cells.npz")
    lam = np.array(idx["lam_cand"]) if lam_from_index else ctrl["lam_cand"]
    return idx["targets"], cells, ctrl, lam


def mixture(dh, dk, ref_path):
    ref = json.load(open(ref_path))["rows"]
    rng = np.random.default_rng(57); worst = 0.0; n = 0
    for ctx, d, from_index in (("hESC", dh, True), ("K562_gw", dk, False)):
        T, cells, ctrl, lam = load(d, from_index)
        tc, own, mu = cells["tc"], cells["own"].astype(float), cells["mu"]
        O, lib = ctrl["O"], ctrl["lib"]
        want = {r["target"]: r for r in ref[ctx]}
        got = 0
        for j, t in enumerate(ctrl["cand"]):
            if lam[j] * lib.mean() < EF.MIN_MU:
                continue
            m = tc == t
            if m.sum() < EF.MIN_CELLS:
                continue
            r = EF.fit_perturbation(own[m], mu[m], O[:, j].astype(float), lam[j] * lib, rng)
            w = want[T[t]]
            d = max(abs(r["e"] - w["nb"]["e"]), abs(r["dbic"] - w["nb"]["dbic"]) / max(1.0, abs(w["nb"]["dbic"])),
                    abs(r["null_dbic"] - w["ctrl_nb"]["dbic"]) / max(1.0, abs(w["ctrl_nb"]["dbic"])))
            worst = max(worst, d); got += 1
        assert got == len(want), (ctx, got, len(want))
        n += got
        print("%s: %d targets refitted" % (ctx, got))
    print("largest difference (e absolute; BIC differences relative): %.2e over %d targets" % (worst, n))
    return worst


def purity(dg, dh, ref_path):
    ref = {r["target"]: r for r in json.load(open(ref_path))["rows"]}
    idx = json.load(open(dg + "/index.json")); T = idx["targets"]
    cand, lam = np.array(idx["cand"]), np.array(idx["lam_cand"])
    ch = np.load(dh + "/ctrl.npz"); O, lib = ch["O"], ch["lib"]
    z = np.load(dg + "/cells.npz")
    tc, own, mu = z["tc"], z["own"].astype(float), z["mu"]
    pur = z["top"] / np.maximum(z["tot"], 1)
    rng = np.random.default_rng(0); worst = 0.0; got = 0
    for j, t in enumerate(cand):
        if lam[j] * lib.mean() < EF.MIN_MU:
            continue
        m = np.where(tc == t)[0]
        if len(m) < 3 * EF.MIN_CELLS:
            continue
        r = EF.fit_perturbation(own[m], mu[m], O[:, j].astype(float), lam[j] * lib, rng, purity=pur[m],
                                tiebreak_rng=np.random.default_rng(t))
        w = ref[T[t]]
        d = max([abs(r["e"] - w["all"])] + [abs(r["e_purity_" + q] - w["purity"][i])
                                            for i, q in enumerate(("low", "mid", "high"))])
        worst = max(worst, d); got += 1
    assert got == len(ref), (got, len(ref))
    print("hESC purity thirds: %d targets refitted; largest difference in e %.2e" % (got, worst))
    return worst


if __name__ == "__main__":
    mode = sys.argv[1]
    worst = (mixture if mode == "mixture" else purity)(*sys.argv[2:5])
    print("PASS" if worst <= TOL else "FAIL")
    sys.exit(0 if worst <= TOL else 1)
