# which-cells-are-measured

Analysis code for the preprint *Escaping cells dilute CRISPR screen responses unequally across cell types* (Xuanrui
Huang, National University of Singapore). Release 1.0.0, archived on Zenodo (https://doi.org/10.5281/zenodo.23480448; all
versions: https://doi.org/10.5281/zenodo.23480447), was made for an earlier, unpublished draft titled *Perturbations
change which cells are measured in single-cell CRISPR screens*. The scripts added or changed since then are listed
under **Version 2** below.

The repository has two parts:

- **`escape_fraction/`**: a small standalone tool that estimates, for each perturbation, the fraction *e* of measured
  cells whose target was not effectively knocked down (the escape fraction), and the dilution (1 − *e*)² it causes (Box 1, recommendation 1). It needs only
  NumPy and SciPy and per-cell counts that any CRISPRi Perturb-seq screen provides.
- **`code/`**: the scripts that produced the numbers in the paper. For release, only paths and imports were changed:
  input and output locations are command-line arguments or come from one configuration file, and the scripts import
  each other from `code/`. No computation was changed (see **Verification**).

## Escape fraction per perturbation

### What it estimates

In a pooled CRISPRi screen, a perturbation's measured cells are a mixture: knocked-down cells, and
escaping cells, in which the target was not effectively knocked down (the guide acted weakly or not at all, the cell was
assigned the wrong guide, or knocked-down cells were depleted while escaping ones remained). Escaping cells carry almost
none of the response, so a response measured on all cells is about (1 − *e*) times the response of the knocked-down
cells as a mean shift, and (1 − *e*)² times as a squared norm or
cross-signal, the magnitude used in the paper. In the paper *e* is about 0.1 for non-essential targets and rises to
0.2–0.5 for the most essential ones, depending on the cell type and the harvest day.

The tool fits each perturbation's per-cell counts of its own target gene with two models, a uniform partial knockdown
and a mixture of knocked-down cells with cells at control level, using a negative-binomial count model whose dispersion is
estimated from control cells. It reports *e* from the mixture, whether the mixture is preferred (BIC difference > 10),
and a null fit on control cells thinned to half their counts, which should not prefer the mixture. The fitting code is
the one that produced the paper's numbers (`code/fate_mixture.py`), copied unchanged.

### Inputs

Two tables, CSV or TSV, optionally gzipped.

**Cells** (one row per perturbed cell):

| Column | Meaning |
|---|---|
| `perturbation` | perturbation name |
| `gene` | the targeted gene whose counts are given; optional, defaults to `perturbation` |
| `count` | raw UMI count of the targeted gene in the cell |
| `expected` *or* `library` | the cell's expected count at control level (control rate of the gene × the cell's library size), or the library size (total UMIs), from which the tool computes it |
| `purity` | optional: UMIs of the assigned guide ÷ all guide UMIs in the cell |

**Controls** (one row per non-targeting control cell and target gene, zero counts included):

| Column | Meaning |
|---|---|
| `gene` | target gene |
| `count` | raw UMI count of the gene in the control cell |
| `expected` *or* `library` | as above; with `library`, the control rate of a gene is the sum of its counts over the sum of the libraries of its control rows |

Use all control cells if possible: the dispersion of each gene and the control rate are estimated from them.

### Running it

```bash
pip install numpy scipy
python escape_fraction/escape_fraction.py cells.csv.gz controls.csv.gz escape.tsv
```

Options: `--min-purity P` keeps only cells whose guide purity is at least *P* before fitting; `--seed N` sets the seed of
the thinned-control null (default 57); `--poisson` replaces the negative binomial (not recommended: with Poisson counts
the null preferred the mixture for 7–13% of genes in the paper's data). The functions can also be imported:
`fit_perturbation(counts, expected, control_counts, control_expected, rng, purity=None)` returns the same fields as a
dictionary.

### Output

One row per perturbation (tab-separated, `NA` where not computed):

| Column | Meaning |
|---|---|
| `n_cells` | cells of the perturbation |
| `e` | escape fraction: share of measured cells at control level |
| `dbic`, `mixture_detected` | BIC(uniform) − BIC(mixture); mixture preferred when > 10 |
| `dilution`, `log2_dilution` | (1 − *e*)², the factor by which escape scales a squared-norm or cross-signal magnitude measured on all cells; its log2 |
| `reliable` | at least 150 cells (see below) |
| `r_kd` | remaining expression of the knocked-down cells, as a fraction of control level |
| `residual` | remaining target expression of all measured cells, Σ count ÷ Σ expected |
| `r_uniform` | knockdown level under the uniform model |
| `nb_size`, `mean_control_expected` | negative-binomial size of the gene in controls; mean expected count per control cell |
| `null_dbic`, `null_mixture` | the same BIC difference for thinned control cells; `TRUE` flags a gene the count model fits poorly |
| `n_purity_*`, `e_purity_low/mid/high` | with a `purity` column: *e* in each third of the cells ranked by guide purity (fitted where a third has ≥ 30 cells) |
| `note` | why a perturbation was not fitted, or what limits its estimate |

A perturbation is fitted when it has at least 30 cells and the gene's mean expected count per control cell is at least
3; with fewer counts per cell the knocked-down and escaping groups cannot be separated. *e* is reported whether or not
the mixture is preferred; when it is not, *e* is small or not identified and the dilution is close to 1.

**How precise is *e*?** In the paper's recovery test (control cells with a known fraction kept and the rest thinned to
20% of their counts; Methods, *Recovery test*), the median error of *e* was at most 0.013 for *e* between 0.1 and 0.7,
with a 90% interval of about ±0.08 at 150 cells, ±0.04 at 400 cells and ±0.1–0.3 at 50 cells. The mixture was detected
for about 80% of perturbations at *e* = 0.1 and 150 cells, and for about half at *e* = 0.9, where few knocked-down cells
remain. The test assumes two populations with a fixed knockdown level; where knockdown varies between cells, the
escaping group also absorbs cells knocked down only partly (in the paper, one population thinned to a knockdown
with a cell-to-cell standard deviation of 0.2 gave *e* of about 0.1).

**Guide purity.** In the hESC screen, most of the baseline *e* came from cells with an uncertain guide call: in the third
of cells with the highest purity, *e* was 0.03–0.05 for non-essential and moderately essential targets (0.16–0.19 over
all cells), while for the most essential targets it stayed at 0.33 (0.48 over all cells), because there it comes from
selection. Restricting to high-purity cells (`--min-purity`, or reading `e_purity_high`) therefore removes
misassignment but not selection.

### Worked example

`escape_fraction/example/` holds a small subset of the K562 genome-scale screen at day 8 (Replogle et al. 2022,
`K562_gwps_raw_singlecell_01.h5ad`): 13 targets chosen by a fixed rule (in each DepMap stratum, the targets at the 25th,
50th and 75th percentile of the paper's *e*, plus the target with the fewest cells) and 10,000 of the 75,328
non-targeting cells. `make_example.py` builds it from the per-cell extraction of `code/k562gw_percell_own.py`.

```bash
python escape_fraction/escape_fraction.py escape_fraction/example/cells.csv.gz escape_fraction/example/controls.csv.gz example_out.tsv
```

runs in a few seconds and prints

```
perturbations 13, fitted 12
mixture detected 10 / 12; thinned-control null 0 / 12
e median 0.150 (quartiles 0.091-0.230); reliable (>= 150 cells) 4
```

`escape_fraction/example/output.tsv` is the expected output. Selected columns, with the paper's estimate from all
control cells (`paper_fit.tsv`):

| Perturbation | DepMap effect | Cells | *e* | BIC difference | Dilution (1 − *e*)² | ≥ 150 cells | *e*, paper |
|---|---|---|---|---|---|---|---|
| RPL21 | −2.74 | 118 | 0.553 | 838 | 0.20 | no | 0.553 |
| RPL26 | −1.94 | 87 | 0.354 | 585 | 0.42 | no | 0.353 |
| RPS21 | −1.62 | 94 | 0.206 | 223 | 0.63 | no | 0.206 |
| ATP5F1D | −0.68 | 159 | 0.301 | 251 | 0.49 | yes | 0.300 |
| PFN1 | −0.66 | 234 | 0.140 | 916 | 0.74 | yes | 0.140 |
| PAICS | −0.52 | 147 | 0.081 | 17 | 0.84 | no | 0.081 |
| AURKAIP1 | −0.42 | 129 | 0.168 | 80 | 0.69 | no | 0.170 |
| UQCRQ | −0.41 | 189 | 0.106 | 65 | 0.80 | yes | 0.107 |
| PRRC2C | −0.24 | 81 | 0.063 | 38 | 0.88 | no | 0.063 |
| SET | −0.11 | 54 | 0.160 | 4 | 0.71 | no | 0.162 |
| PRDX6 | −0.06 | 228 | 0.094 | 203 | 0.82 | yes | 0.094 |
| HBZ | −0.03 | 71 | 0.048 | 2 | 0.91 | no | 0.047 |
| PARK7 | — | 8 | not fitted (fewer than 30 cells) | | | | |

Reading it: for RPL21, a ribosomal protein that every line depends on, more than half of the cells measured at day 8
had escaped knockdown, so a response computed from all its cells is about one fifth of the knocked-down-cell
response in squared-norm units. For SET and HBZ the mixture is not preferred (BIC difference < 10), so their *e* is not
evidence of escape. The small differences from the paper's column come from using 10,000 rather than all control cells
for the dispersion and the control rate; on the full data the tool reproduces the paper's fits exactly (see
**Verification**).

## Analysis scripts

### Configuration

Scripts take their inputs and outputs as command-line arguments (each script's docstring ends with its usage). The few
locations that were fixed in the scripts as they ran are read from `config.env` at the repository root, or from
environment variables of the same name (`code/config.py`). Copy `config.example.env` to `config.env` and set:

| Key | Meaning |
|---|---|
| `DEPMAP` | DepMap Public 24Q4 `CRISPRGeneEffect.csv`; the median gene effect over lines defines the essentiality strata |
| `FATE_ROOT` | output tree of the hESC, K562, RPE1 and CD4 analyses (`budget/`, `budget_cd4/`, `budget_d8/`, `kdgroup/`, `samescale_out/`, `pan_libnorm/`, `phase_decomp/`, `xctx/`, `ci/`, ...) |
| `CC_ROOT` | output tree of the Jurkat / HepG2 budget, the Maffa, time-course and VCC H1 analyses, the fate table and `libshare/` |
| `PAN_KDGROUP` | hESC per-cell knockdown-group aggregates (`pan_extract_kdgroup.py`) |
| `PAN_GUIDE_RAW` | hESC aggregates by guide parity with raw counts (`pan_extract_guide.py`) |
| `ORION_ROOT` | X-Atlas/Orion download and its preprocessed gene axis (`orion_extract.py`) |
| `CD4_PERT_TABLE`, `CD4_PSEUDOBULK` | CD4 donor 1 resting inputs of `fate_libshare_summary.py` (see **Inputs made outside this repository**) |

The analyses ran on two machines whose output trees reuse folder names, so they are kept as two roots.

Environment: Python ≥ 3.10 with NumPy, SciPy, h5py, pandas, PyArrow and Matplotlib (`requirements.txt`). The paper used
Python 3.11 with NumPy 2.4 and SciPy 1.17; the escape-fraction tool was also run with NumPy 2.2 and SciPy 1.15.

### Scripts by figure

Extraction scripts read the raw public files and write per-target or per-cell arrays; analysis scripts read those
arrays. Each script's docstring states its question, its inputs and its rules, and ends with its usage.

| Item | Extraction | Analysis |
|---|---|---|
| Fig. 1: depletion and arrest across contexts; Supplementary Table S1 | `pan_extract.py`, `pan_extract_guide.py`, `fate_budget_extract.py`, `orion_extract.py` | `pan_fate.py`, `feng_fate.py`, `fate_h1.py`, `fate_iu_cd4.py`, `fate_iu_pan.py`, `fate_maffa.py`, `fate_orion_c1.py`, `fate_leng.py`, `pan_library_norm.py` → `fate_c1_table.py` |
| Fig. 2: two-component fits; Supplementary Table S2 | `pan_percell_own.py`, `k562gw_percell_own.py`, `cd4_percell_own.py`, `pan_percell_guideconf.py` | `fate_mixture.py`, `fate_mixture_more.py` (further contexts), `fate_rev_mixture.py` (two guides of one target, the two K562 screens, level of the silenced cells); recovery test `fate_mixture_calib.py`; guide purity `fate_mixture_guideconf.py` |
| Fig. 3: high-residual share and dilution | (Fig. 2, Fig. 4 and Supplementary Fig. S4 extractions) | `fate_dilution_curve.py`, `fate_rev_dilution.py`, `fate_k562_repeat.py` |
| Fig. 4: hESC–K562 comparison; Supplementary Tables S3, S4 | `pan_extract_guide.py`, `k562_samescale_extract.py`, `pan_extract_kdgroup.py`, `pan_extract_mixnull.py` | `fate_samescale.py`, `fate_dilution_model.py`, `fate_kdgroup.py`, `fate_selection_check.py`, `fate_rev_samescale.py` (symmetric silenced comparison, estimators, abundance bins), `fate_rev_mixnull.py` (mixed classification control) |
| Fig. 5a: direction with silenced cells | `pan_percell_expr.py` | `fate_c34_agree.py`, `fate_rev_dir_orion.py` |
| Fig. 5b and Supplementary Fig. S4a–d: composition, within-cell change and selection; Supplementary Table S5 | `fate_budget_extract.py`, `pan_budget_extract.py`, `cd4_budget_extract.py`, `phase_decomp_extract.py`, `pan_extract_state.py` | `fate_budget.py`, `fate_budget_screens.py`, `phase_decomp.py`, `fate_rev_comp.py` (symmetric share and bounds), `fate_rev_state.py` (composition by pluripotency state in hESC) |
| Fig. 5c and Supplementary Fig. S4e: composition removal; null-perturbation control | `fate_vectors_export.py` | `fate_crosscontext.py`, `fate_c6_null.py` |
| Fig. 5d–e: benchmark, raw and denoised; Supplementary Table S6 | `fate_vectors_export.py` | `fate_benchmark.py`, `fate_benchmark_denoised.py` |
| Supplementary Fig. S1: controls for the two-component test | (Fig. 2 extractions) | `fate_mixture_controls.py` |
| Supplementary Fig. S2: time course | `fate_budget_extract.py` (with `FATE_TARGET_LIST` for the day-8 screen) | `fate_drepanos.py`, `fate_budget_screens.py`, `feng_time_response.py`, `fate_maffa.py` |
| Supplementary Fig. S3: two further published comparisons | `fate_vectors_export.py`, `orion_extract.py` | `fate_nadig_pairs.py`, `fate_orion_matched.py`, `fate_orion_matched2.py` |
| Supplementary Fig. S4f: library share of heavy dilution | `fate_libshare_k562gw.py` | `fate_libshare_summary.py` |
| Supplementary Fig. S5: depletion beyond dependency | `fate_vectors_export.py` | `fate_priming_maffa.py`, `audit_bq_floor.py`, `fate_jost.py`, `fate_c7_programs.py`, `fate_c7_posthoc.py` |
| Supplementary Figs. S6–S7: phase calls, per-cell classification | (Fig. 4 extractions) | `fate_suppdata.py`, `fate_kdgroup.py` |
| Supplementary Fig. S8: intervals | — | `fate_ci.py`, `fate_ci_5090.py` |
| Figures | `fate_figdata_a100.py`, `fate_figdata_5090.py`, `fate_suppdata.py` collect the outputs | `plot_fate_figs_v6.py` (Figs. 1–5), `plot_fate_supp_v6.py` (Supplementary Figs. S1–S8); both take helpers from `plot_fate_figs_v2.py`, which with `plot_fate_figs.py` and `make_fate_supplement.py` drew earlier versions |

The figure scripts read only saved outputs, copied unchanged into four folders given on the command line
(`plot_fate_figs_v6.py <figdata> <v1 figdata> <v3 figdata> <v4 figdata> <out>`): `figdata` holds the files listed in the
docstring of `plot_fate_figs_v2.py`; `v1 figdata` the JSON files written by `fate_figdata_*.py` and `fate_suppdata.py`;
`v3 figdata` the metrics of `fate_mixture_controls.py`, `fate_nadig_pairs.py`, `fate_orion_matched2.py`,
`fate_k562_repeat.py` and `fate_selection_check.py`; `v4 figdata` the output folders of the `fate_rev_*.py` scripts and
`fate_mixture_more.py`, with the two `fate_rev_comp.py` outputs named `rev_comp_a100.json` and `rev_comp_5090.json`.

Shared modules: `strata.py` (DepMap essentiality and strata, cell-cycle and p53 gene sets), `screen_io.py` (readers for the
Replogle and Nadig h5ad files) and `config.py`. `strata.py` and `screen_io.py` hold functions that, during the
analysis, lived in modules of another project; they were lifted unchanged.

### Data

All data are public; accessions are in the Data availability section of the preprint.

| Context | Source |
|---|---|
| K562 (essential, day 6; genome-scale, day 8) and RPE1 (day 7) | Replogle et al. 2022, figshare (`K562_essential_raw_singlecell_01.h5ad`, `K562_gwps_raw_singlecell_01.h5ad`, `rpe1_raw_singlecell_01.h5ad`) |
| Jurkat and HepG2 (day 7) | Nadig et al. 2025, GEO GSE264667 |
| Primary CD4+ T cells, four donors | Zhu et al. 2026, GEO GSE314342 |
| hESC, genome-scale CRISPRi | Pan et al. 2026, GEO GSE295214; library composition from its Supplementary Table 6 |
| iPSC, 19 lines, days 3–5 | Feng et al. 2025, figshare |
| HCT116 and HEK293T (X-Atlas/Orion) | Hugging Face `Xaira-Therapeutics/X-Atlas-Orion` |
| H1 hESC | Virtual Cell Challenge 2025 datasets |
| 16 cancer lines, Cas9 knockout, day 6 | figshare item 33273600 |
| K562 time course, CRISPRi and knockout, days 4–14 | Zenodo record 20722064 |
| K562 mismatched-guide dose titration | GEO GSE132080 |
| iPSC-derived astrocytes | GEO GSE182308 |
| Essentiality | DepMap Public 24Q4 `CRISPRGeneEffect.csv` |

### Inputs made outside this repository

Two analyses read per-target tables made by scripts of another project in the same working repository:
`fate_c1_table.py` reads the K562, RPE1, Jurkat, HepG2 and CD4 donor-1 tables under `CC_ROOT/explore_W/`, and
`fate_libshare_summary.py` reads the CD4 donor-1 resting table and pseudobulk (`CD4_PERT_TABLE`, `CD4_PSEUDOBULK`).
They hold, per target, the cell number, the knockdown residual, the G1 change and the response energy, defined as in
`pan_fate.py`. `orion_extract.py` reads the X-Atlas/Orion gene axis and control means from a preprocessed copy of the
Orion data (`ORION_ROOT/PIE_x_atlas_orion/preprocessed/`: `meta.json`, `ctrl_means.npy`) made in the same project.

### Verification

On 2026-10-19 one analysis per figure, the three checks of the mixture estimator and the escape-fraction tool were
rerun from this repository on the servers where the data sit, with the configuration above, and every output was
compared value by value with the output the paper's numbers were taken from (`checks/compare_json.py`; relative
tolerance 10⁻⁹, ignoring only the stored docstring). On 2026-10-20 every script added for the final version of the
earlier draft was rerun the same way, with the arguments of its original run, and the figure scripts redrew Figures 1–5
and Supplementary Figures S1–S8 from the saved outputs (PNG pixel-identical; PDF identical apart from the creation
date). The results are in `checks/README.md`.

## Version 2

The scripts below were added or changed for the preprint after release 1.0.0 (2026-10-26). Figure and table numbers in
this section are those of the preprint; the table under **Scripts by figure** gives those of the earlier draft.

| Item (version 2) | Script | What it adds |
|---|---|---|
| Fig. 5: transfer model and benchmark; Supplementary Table S5 | `fate_vectors_export.py` (changed), `fate_benchmark.py`, `fate_benchmark_denoised.py` (changed), `fate_transfer_coef.py` (new), `fate_ci_bench_guidekd.py` (new); driver `run_v2_guidekd.sh` | the hESC knocked-down ground truth is now taken on guide halves (`kd_g`), like the all-cell ground truth; single-coefficient transfer model; intervals for the benchmark |
| Supplementary Table S2, further checks 1 and 3 | `fate_v2_further_checks.py` (new) | escape fraction against cell number within strata; change in direction with the classification control subtracted |
| Supplementary Table S2, further check 4 | `fate_v2_kd_depth.py` (new) | hESC–K562 ratio at similar knockdown depth |
| Supplementary Table S2, further check 5 | `fate_drepanos_escape.py` (new) | per-day escape fraction in the K562 time course (not fittable: too few counts per cell) |
| Supplementary Table S2, further check 2 | `fate_kdgroup.py` (unchanged; keys `Q1`, `Q1p`) | response of escaping cells relative to knocked-down cells |
| Supplementary Data 1 | `make_supp_data1.py` (new) | escape fraction of every fitted perturbation, from the outputs of `fate_mixture.py` and `fate_mixture_more.py` |
| Figures | `plot_fate_figs_v7.py` (Figs. 1–6), `plot_fate_supp_v7.py` (Supplementary Figs. S1–S12) | |

`plot_fate_figs_v7.py <figdata> <v1 figdata> <v3 figdata> <v4 figdata> <out>` reads the folders described above, plus, in
`<v4 figdata>`: `transfer_coef.json` (from `fate_transfer_coef.py`) and `bench_guidekd/` holding the four
`w5_benchmark_denoised_<source>.json` files and `ci_bench.json` written by `run_v2_guidekd.sh`.
`plot_fate_supp_v7.py` takes the same arguments.

**Verification of version 2.** The extraction behind Fig. 5 had to be rerun (`run_v2_guidekd.sh`), because the arrays
of 2026-10-18 had been removed. The rerun reproduced the K562 and RPE1 count arrays exactly; for hESC the cell numbers of
every group were identical and only the library sums of single cell-cycle stages differed (largest difference 1,426
UMIs), and the all-cell benchmark values were reproduced to the stored precision. The figure scripts of this repository
redrew all six figures and twelve supplementary figures pixel-identically to those of the preprint, and
`make_supp_data1.py` wrote Supplementary Data 1 byte-identically. `fate_v2_further_checks.py` reproduces the values of
Supplementary Table S2 exactly. The new analysis scripts ran once on the servers; unlike those of release 1.0.0, they were not rerun
from this repository.

## Notes

- Comments and `rules` strings carry labels such as "findings bm", "C5 W5", "redesign §5.3" or "check V3". They refer to
  the authors' analysis log and are kept so that each script can be traced to the record it was written for. Figure
  numbers in docstrings are those of the draft the script was written for; the tables above give the preprint's.

## Licence

MIT (see `LICENSE`).

Contact: huangxuanrui@u.nus.edu
