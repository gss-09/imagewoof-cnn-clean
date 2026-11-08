# Round 3: stronger augmentation, second seed

Completed November 7–8, 2026, source SHA-256 28607dec… (adds opt-in `--rotate`; default 0 reproduces earlier runs).
Both runs: DogCNN-v4, 150 epochs, AdamW lr 0.001 wd 0.001, batch 32, label smoothing 0.1, EMA 0.999 saved, workers 6.

Reference: `r2_v4_long150` (crop 0.35, seed 42), 83.47% (1,081/1,295), flip 84.40% (1,093).

| Run | Change | EMA best (saved) | Public `predict` | Flip-averaged (separate script) | Raw best | Final train |
| --- | --- | --- | --- | --- | --- | --- |
| `r3_v4_aug_strong` | crop 0.25 + rotation ±10° | 82.63% @88 | 82.63% (1,070) | 83.47% (1,081) | 82.01% @117 | 98.18% |
| `r3_v4_seed43` | seed 43 only | 82.93% @53 | 82.93% (1,074) | 83.32% (1,079) | 82.86% @103 | 99.57% |

Findings:

- Seed noise: an identical recipe with another seed scored 0.54 point (7 images) lower. The 83.47% best is partly a lucky draw; the recipe's true level is roughly 83 ± 0.5.
- Stronger augmentation lowered train accuracy (98.2% vs 99.7%) but also lowered validation by 0.8 point: crop 0.35 without rotation remains best.
- Flip averaging again added +0.4 to +0.9 point.
- No new best; keep `saved_models/v4_long150_8347/`. Weights are not tracked in Git.
