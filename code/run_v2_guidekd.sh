#!/bin/bash
# Version 2 (2026-10-26): score the hESC knocked-down ground truth on guide halves, like the all-cell ground truth.
# The extraction is rerun with the same scripts, then the vectors (now with kd_g), the denoised benchmark for each source,
# the transfer coefficients and their intervals. Locations from the environment (see config.example.env):
#   DEPMAP, FATE_ROOT (the original budget/ counts, for the comparison), PAN_GUIDE_RAW, REPLOGLE_DIR (folder with
#   K562_essential_raw_singlecell_01.h5ad and rpe1_raw_singlecell_01.h5ad), PAN_RAW (hESC raw files), OUT (new folder),
#   and the Jurkat and HepG2 vectors copied into $OUT/xctx from CC_ROOT/xctx (made on the other machine).
# Run from the repository root.
set -u
C=code; export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p $OUT/budget $OUT/xctx $OUT/logs
run() { n=$1; shift; s=$(date +%s); python3 "$@" > $OUT/logs/$n.log 2>&1; echo "$n exit=$? $(( $(date +%s) - s ))s" >> $OUT/logs/STATUS; }
( run extract_k562_rpe1 $C/fate_budget_extract.py $DEPMAP $OUT/budget K562:$REPLOGLE_DIR/K562_essential_raw_singlecell_01.h5ad RPE1:$REPLOGLE_DIR/rpe1_raw_singlecell_01.h5ad ) &
( run extract_hesc $C/pan_budget_extract.py $PAN_RAW $PAN_GUIDE_RAW $DEPMAP $OUT/budget 12 ) &
wait
run vectors $C/fate_vectors_export.py $OUT/budget $OUT/xctx K562 RPE1 hESC
for s in K562 Jurkat HepG2 RPE1; do run bench_$s $C/fate_benchmark_denoised.py $DEPMAP $OUT/xctx $s hESC; done
run transfer $C/fate_transfer_coef.py $DEPMAP $OUT/xctx hESC $OUT/transfer_coef.json K562 Jurkat HepG2 RPE1
run ci_bench $C/fate_ci_bench_guidekd.py $DEPMAP $OUT/xctx $OUT/ci_bench.json
echo ALLDONE >> $OUT/logs/STATUS
