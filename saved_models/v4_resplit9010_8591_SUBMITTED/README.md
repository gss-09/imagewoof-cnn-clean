# SUBMITTED MODEL: DogCNN-v4, 86.72% recipe on a 90/10 re-split, 85.91% held-out

Submitted to Canvas by the user on November 8, 2026 (the TA said the epoch must be the one with the highest validation score).
From `runs/r8_v4_resplit9010`, seed 42. Supplied train+val pooled (10,360), re-split 90/10 per breed: 9,324 train / 1,036 held out (`split.json`).
Recipe: DogCNN-v4, SGD Nesterov 0.9, lr 0.05, wd 0.001, warm-up 5 + cosine 150, batch 32, crop 0.35–1.0, flip, jitter 0.15, label smoothing 0.1, dropout 0.3, EMA 0.999.
EMA best epoch 83: **85.91% (890/1,036)**, single view. Reproduce: `python3 saved_models/v4_resplit9010_8591_SUBMITTED/eval_split.py .`
This bundle is the run checkpoint; compare its SHA-256 in manifest.json with the model.pth inside the submitted zip.
