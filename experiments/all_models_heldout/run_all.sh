#!/bin/bash
# Run predict_all.py for every run with weights; use the run's own solution.py if present, else the root one.
cd "$(dirname "$0")/../.." || exit 1
OUT=${EVAL_OUTPUT_DIR:-runs/alleval}
mkdir -p "$OUT" || exit 1
for d in runs/*/; do r=$(basename $d); [ -f $d/model.pth ] || continue
  code=$d; [ -f $d/solution.py ] || code=.
  python3 experiments/all_models_heldout/predict_all.py $code $d/model.pth . $OUT/pred_$r.json > /dev/null 2>$OUT/err_$r.txt && echo "ok $r" || echo "FAIL $r ($code)"
done
