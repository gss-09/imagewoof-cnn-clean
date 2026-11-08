# Round 9: 86.72% recipe on all train+val (100/0), full 150 epochs

Launched November 8, 2026, source SHA-256 ad63459a…, `--train-on-val`, seed 42, final epoch (150) saved.
Trains on all 10,360 supplied images; no images held out, so there is **no honest accuracy** (in-training "validation" 100% is memorized; train under augmentation 99.64%).
Checkpoint SHA-256 c7a9e19f… (verified); CPU verifier passed.

Differs from `runs/r6_v4_trainval_ep125` only in stopping at 150 instead of 125. Expected hidden-test gain over the 86.72% model is small (the split trend 70/30 83.92% → 90/10 85.91% → supplied 86.72% suggests roughly +0.5–1.5 points from 14% more data), but it cannot be verified.
