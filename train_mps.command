#!/bin/zsh
# Run by double-clicking in Finder or invoking from a normal macOS Terminal.
set -euo pipefail
TASK_DIR="${0:A:h}"
cd "$TASK_DIR"
PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi
RUN_DIR="runs/mps_full_$(date +%Y%m%d_%H%M%S)"
print "Using Python: $PYTHON_BIN"
print "Training output: $TASK_DIR/$RUN_DIR"
print "MPS is required; this command will not fall back to CPU."
# macOS caffeinate prevents idle sleep while the training process runs.
caffeinate -i "$PYTHON_BIN" -u solution.py --device mps --epochs 50 --output "$RUN_DIR"
print "Training completed. Checking checkpoint portability on CPU..."
"$PYTHON_BIN" verify_pipeline.py "$RUN_DIR/model.pth"
