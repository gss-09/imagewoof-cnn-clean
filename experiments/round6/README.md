# Round 6: weight decay 0.002, and a train + validation retrain

Launched November 8, 2026 (source SHA-256 327e43fb…). Base recipe: the 86.72% run `r5_v4_sgd_wd001` (DogCNN-v4, SGD Nesterov lr 0.05, 5 warm-up epochs + cosine over 150, crop 0.35–1.0, EMA 0.999, seed 42).

Code change: `--train-on-val` adds the validation images to training (with training augmentation) and saves the latest epoch; `--stop-epoch N` ends training at epoch N while the schedule still spans `--epochs`. Defaults unchanged (CPU smoke runs of both modes passed).

| Run | accelerator | Change | Result |
| --- | --- | --- | --- |
| `r6_v4_sgd_wd002` | 0 | wd 0.001 → 0.002 | EMA best 85.64% (1,109/1,295) @113; final epoch 84.63%, train 99.5% |
| `r6_v4_trainval_ep125` | 1 | train on train+val, stop at epoch 125 | No held-out score possible; 99.92% on its own training images; CPU verifier passed |

Findings:

- wd 0.002 slowed learning (73% vs 81% at epoch 48), caught up near epoch 110, peaked at 113, then still memorized. 14 images below wd 0.001, so 0.001 stays.
- The train+val retrain cannot be evaluated on held-out data. It is a candidate final model only if the group accepts submitting an unscored model; the measured best remains `saved_models/v4_sgd_wd001_8672/` (86.72%).
