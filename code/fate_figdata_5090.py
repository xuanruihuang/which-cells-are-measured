"""Export the fate paper's figure data held on the 5090 into one JSON (2026-10-18): the C1 table, Jurkat / HepG2
budget, Maffa 16 lines, the Drepanos time course, the VCC H1 signatures and the 5090 intervals.
Usage: fate_figdata_5090.py <out json>."""
import json, os, sys
import config

R = config.get("CC_ROOT")


def grab(path, drop=("rules",), keep=None):
    if not os.path.exists(path):
        return None
    m = json.load(open(path))
    if keep:
        return {k: m[k] for k in keep if k in m}
    for k in drop:
        m.pop(k, None)
    return m


def main():
    out = {"rules": __doc__}
    out["c1_table"] = grab(R + "/c1_table/c1_table.json")
    out["ci_5090"] = grab(R + "/ci/fate_ci_5090.json")
    b = grab(R + "/budget/metrics_fate_budget.json")
    out["budget"] = {lab: {k: v[k] for k in ("meta", "random", "guide", "random_no_cycle_genes", "negative_control",
                                             "T3_calibration") if k in v}
                     for lab, v in (b or {}).items() if lab in ("Jurkat", "HepG2")}
    M = grab(R + "/fate_maffa/metrics_fate_maffa.json")
    out["maffa"] = dict(per_line=M["per_line"], M2=M["M2"], M3={k: v for k, v in M["M3"].items() if k != "per_gene"},
                        M3_per_gene=M["M3"]["per_gene"], response_by_depletion_bin=M["response_by_depletion_bin"],
                        rows=M["rows"], null=M["null"], M1=M["M1_spearman_depletion_effect"]) if M else None
    out["priming_maffa"] = grab(R + "/priming_maffa/metrics_priming_maffa.json")
    out["drepanos"] = grab(R + "/fate_drepanos/metrics_fate_drepanos.json")
    out["h1"] = grab(R + "/fate_h1/metrics_fate_h1.json")
    json.dump(out, open(sys.argv[1], "w"))
    print("written", sys.argv[1], "keys", [k for k in out if k != "rules"], flush=True)


if __name__ == "__main__":
    main()
