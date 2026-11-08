# DogCNN-v4, 86.72% recipe on a 95/5 re-split: 87.84% held-out

From `runs/r10_v4_resplit9505` (Nov 8, 2026), seed 42. Supplied train+val pooled (10,360) and re-split 95/5 per breed: 9,842 train / 518 held out (`split.json`).
Recipe identical to `saved_models/v4_sgd_wd001_8672/` (DogCNN-v4, SGD Nesterov 0.9, lr 0.05, wd 0.001, warm-up 5 + cosine 150, batch 32, crop 0.35–1.0, flip, jitter 0.15, label smoothing 0.1, dropout 0.3, EMA 0.999).
EMA best epoch 126: **87.84% (455/518)**, single view. Reproduce: `python3 saved_models/v4_resplit9505_8784/eval_split.py .` from the project root.
Note: on the original 1,295 supplied validation images it scores ~99% because ~95% of them are its training images; the honest number is the 518 held-out images.
