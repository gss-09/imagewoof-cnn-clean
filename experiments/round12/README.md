# Round 12: 86.72% recipe on a seeded, stratified 99/1 re-split

Launched November 8, 2026, source SHA-256 ad63459a…, `--resplit 0.01`, seed 42, 150 epochs → 10,255 train / 105 validation.

- EMA best (saved) epoch 113: **89.52% (94/105)**, reproduced by public `predict` and on CPU by `saved_models/v4_resplit9901_8952/eval_split.py`. Final epoch 84.76%. Checkpoint SHA-256 2d8f6073… (verified).
- Per breed: shih_tzu 73, english_foxhound 67 (6 images), old_english_sheepdog 100, samoyed 100, others ~91.
- 105 images: one image ≈ 0.95 point, so this is a very noisy estimate. Bundle: `saved_models/v4_resplit9901_8952/`, tag `v4-resplit9901-89.52`.
