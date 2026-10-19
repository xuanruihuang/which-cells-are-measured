"""DepMap essentiality strata and the cell-cycle gene sets, shared by the analyses.
Lifted unchanged from the modules that held them during the analysis: essentiality() and TIERS from fate_iu_cd4.py
(the same function as fate_budget.essentiality), essentiality_of() from pan_fate.py, the Tirosh S / G2M gene lists
from tahoe_extract.py and explore_V_day6_day8.py (identical copies), and the p53 target list P53 from tahoe_extract.py.
The only change is that the DepMap table is an argument, taken from the DEPMAP setting (config.py) when not given."""
import numpy as np
import config

TIERS = (("lt_-1", -np.inf, -1.0), ("-1_-0.5", -1.0, -0.5), ("-0.5_-0.2", -0.5, -0.2), ("gt_-0.2", -0.2, np.inf))
S_GENES = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1",
           "UHRF1", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3",
           "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1",
           "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
G2M = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO",
       "CENPF", "TACC3", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1",
       "KIF20B", "HJURP", "CDCA3", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8",
       "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3",
       "CBX5", "CENPA"]
P53 = ["MDM2", "FDXR", "TP53I3", "RRM2B", "ZMAT3", "BAX", "CDKN1A", "GDF15", "SESN1", "SESN2", "TNFRSF10B", "AEN",
       "PLK3", "TP53INP1", "DDB2", "XPC", "RPS27L", "TRIAP1", "FAS", "BBC3", "PHLDA3", "SPATA18", "POLH", "TIGAR",
       "CCNG1", "APAF1", "EDA2R", "SULF2"]


def essentiality(path=None):
    """Median DepMap gene effect over all lines, per gene symbol (CRISPRGeneEffect csv, header "SYMBOL (ID)")."""
    with open(path or config.get("DEPMAP")) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        rows = [np.array([float(x) if x else np.nan for x in line.rstrip("\n").split(",")[1:]]) for line in f]
    return dict(zip(sym, np.nanmedian(np.array(rows), 0)))


def essentiality_of(want, path=None):
    """The same medians, read only for the genes in `want`."""
    with open(path or config.get("DEPMAP")) as f:
        sym = [h.split(" (")[0] for h in f.readline().rstrip("\n").split(",")[1:]]
        cols = [i for i, s in enumerate(sym) if s in want]
        rows = []
        for line in f:
            v = line.rstrip("\n").split(",")[1:]
            rows.append([float(v[i]) if v[i] else np.nan for i in cols])
    med = np.nanmedian(np.array(rows), 0)
    return {sym[i]: m for i, m in zip(cols, med)}
