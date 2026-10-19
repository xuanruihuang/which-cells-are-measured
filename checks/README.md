# Verification of the release

Run on 2026-10-19 from this repository (branch `release`), on the two servers that hold the data, with the locations set
through `config.env` / environment variables as described in the main README. Each rerun wrote to a new folder; for the
two scripts that write next to their inputs (`fate_c6_null.py`, `fate_benchmark_denoised.py`) and for
`fate_libshare_summary.py`, the inputs were linked into a new folder so that no original output was touched.

`compare_json.py <original> <rerun>` compares every value of the two outputs (relative tolerance 10⁻⁹; the `rules`
docstring is not compared). "Identical" below means every value of the original output is reproduced. Items are
numbered as in the preprint.

## One analysis per figure

| Item | Script rerun | Values compared | Result | Values seen in the rerun |
|---|---|---|---|---|
| Fig. 1 | `fate_orion_c1.py` | 50 | identical | dG1, most essential: HCT116 0.094, HEK293T 0.002 |
| Fig. 1, Supplementary Table S1 | `fate_c1_table.py` | 346 | identical | dG1, most essential: RPE1 0.44, HepG2 0.26, K562 0.05, Jurkat 0.05; CD4 residual 0.78 |
| Fig. 4 | `fate_samescale.py` | 103 | identical; the rerun adds 3 fields | hESC / K562 0.41 (all targets), 1.18 (knockdown good on both sides; 353 matched, 347 positive), published-style 0.34; its interval comes from `fate_ci.py` |
| Fig. 2 | `fate_mixture.py` | see below | see below | |
| Supplementary Figs. S2, S4 | `fate_budget_screens.py` | 16,430 | identical | K562 residual day 6 / day 8 0.21 / 0.33; across screens composition 0.0058, selection 1.95 |
| Fig. 5b, Supplementary Fig. S4 | `fate_budget.py` (K562, RPE1, hESC) | 65,547 | identical; the rerun adds 12 fields | composition share RPE1 0.245 (n = 663), K562 0.0040, hESC 0.0035; selection hESC 2.80, K562 1.56, RPE1 1.46 |
| Fig. 5b, Supplementary Fig. S4 | `fate_budget.py` (Jurkat, HepG2) | 39,442 | identical; the rerun adds 8 fields | composition share HepG2 0.090, Jurkat 0.0082; selection Jurkat 1.74, HepG2 1.46 |
| Supplementary Fig. S4f | `fate_libshare_summary.py` | 129 | identical | heavy distortion, most essential / non-essential: hESC 0.665 / 0.136, K562 0.369 / 0.108, CD4 0.941 / 0.244 |
| Fig. 3 | `fate_dilution_curve.py` | 82 | identical | hESC de-attenuated slope 1.05 [0.77, 1.43], n = 109, ρ 0.78, reliability 0.87; CD4 ρ 0.32 / 0.19 / 0.28 |
| Fig. 5a | `fate_c34_agree.py` | 39 | identical | most essential ρ 0.577 / 0.573 (n = 41), null 97.5% 0.054; all strata Δ −0.044 |
| Fig. 5c | `fate_c6_null.py` | 28 | identical | RPE1 Δρ own composition removed +0.074; non-essential donor ≤ 0.005 |
| Fig. 5d–e | `fate_benchmark_denoised.py` (K562 source) | 321 | identical | scale factor 0.48 → 0.85, relative error 3.13 → 1.32, noise share 0.73 |
| Supplementary Fig. S5 | `fate_jost.py` | 906 | identical | dilution-index heterogeneity p = 0.87 (25 genes) |

The added fields: `fate_samescale.py` and `fate_budget.py` were extended (an interval for the matched ratio; a
guide-design negative control) after the stored outputs were written, so the rerun writes a few fields the stored
outputs do not have. Every field present in both is identical.

## Mixture estimator (Fig. 2, Supplementary Table S6; redesign notes §5.3, §6.4–6.6)

| Check | Script | Values compared | Result | Values seen |
|---|---|---|---|---|
| Fig. 2 fits, hESC and K562 day 8 | `fate_mixture.py` | 16,828 | identical | analysable 362 / 436; mixture preferred (NB) by stratum hESC 0.63 / 0.89 / 0.97 / 0.93, K562 0.88 / 0.93 / 0.86 / 0.81; thinned-control null at most 0.018; median e most essential → non-essential hESC 0.53 → 0.17, K562 0.35 → 0.09 |
| Recovery test (§6.4) | `fate_mixture_calib.py` | 15,300 | identical | n = 150: median bias ≤ 0.013 for e 0.1–0.7, 90% interval about ±0.08; detection at e = 0.9: hESC 53%, K562 42% |
| Guide purity (§6.5) | `fate_mixture_guideconf.py` | 4,723 | identical | median e by purity third (low / mid / high) and all cells: DepMap > −1, 0.32 / 0.09 / 0.04, all 0.18 (high ÷ all 0.24); most essential 0.58 / 0.35 / 0.33, all 0.48 (0.69) |
| Five further contexts (§6.6) | `fate_mixture_more.py` | 33,218 | identical | mixture preferred, most essential: K562 day 6 0.82, RPE1 0.79, Jurkat 0.91, HepG2 0.75, CD4 0.75; thinned-control null at most 0.012; median e most essential / non-essential: K562 day 6 0.19 / 0.10, RPE1 0.14 / 0.07, Jurkat 0.29 / 0.09, HepG2 0.11 / 0.12, CD4 0.17 / 0.16 |

## escape_fraction.py against the paper's fits

`check_escape_fraction.py` refits with `escape_fraction.fit_perturbation` and compares target by target.

| Check | Targets | Largest difference | Result |
|---|---|---|---|
| e, BIC difference and thinned-control BIC difference vs `fate_mixture.py` (hESC 362, K562 day 8 436) | 798 | 0 | pass |
| e for all cells and per guide-purity third vs `fate_mixture_guideconf.py` (hESC) | 292 | 0 | pass |

The worked example was also run with `expected` columns instead of `library` (same *e* in every row) and with a
`purity` column and `--min-purity`.

## Scripts added for the final version of the preprint

`fate_rev_*.py`, `pan_extract_mixnull.py`, `pan_extract_state.py`, `fate_mixture_controls.py`, `fate_nadig_pairs.py`,
`fate_orion_matched*.py`, `fate_k562_repeat.py`, `fate_selection_check.py` and `orion_extract.py` were copied into this
repository on 2026-10-20 with only their paths and imports changed (server `sys.path` inserts removed; fixed locations
read through `config.py`; `orion_extract.py` takes its gene sets from `strata.py`, whose lists are identical to those it
imported). They have not been rerun from this repository. Their stored outputs are the inputs of the figure scripts,
and `plot_fate_figs_v6.py` and `plot_fate_supp_v6.py` run from this repository redrew Figures 1–5 and Supplementary
Figures S1–S8 identically (PNG pixel-identical; PDF identical apart from the creation date).
