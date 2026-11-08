# DogCNN-v4, 86.72% recipe on a 99/1 re-split: 89.52% held-out

From `runs/r12_v4_resplit9901` (Nov 8, 2026), seed 42. Supplied train+val pooled (10,360) and re-split 99/1 per breed: 10,255 train / 105 held out (`split.json`).
Recipe identical to `saved_models/v4_sgd_wd001_8672/` (DogCNN-v4, SGD Nesterov 0.9, lr 0.05, wd 0.001, warm-up 5 + cosine 150, batch 32, crop 0.35–1.0, flip, jitter 0.15, label smoothing 0.1, dropout 0.3, EMA 0.999).
EMA best epoch 113: **89.52% (94/105)**, single view. Reproduce: `python3 saved_models/v4_resplit9901_8952/eval_split.py .` from the project root.
Caution: 105 images means one image ≈ 0.95 point; the score is very noisy. Final epoch was 84.76%.
