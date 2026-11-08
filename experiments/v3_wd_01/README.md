# DogCNN-v3, AdamW weight decay 0.01

Hypothesis: raising weight decay 0.0001 -> 0.001 gave +12 images (80.93%) while train accuracy still reached ~98%; stronger decoupled decay (0.01) may reduce overfitting further. Only change versus `runs/v3_weight_decay_001`: weight decay 0.01. Fresh 50-epoch run, seed 42, supplied split. Log: `setup/v3_wd_01.log`.

Status: launched and completed November 7, 2026.

## Result

Completed all 50 epochs. Best epoch 39: **79.4595% (1,029/1,295)**, reproduced through public `predict` after reload; 19 images fewer than the 0.001 run (80.93%). Final epoch 50: train ~97.9%, validation ~78.8%. Flip averaging, measured separately on CPU with a scratch script (not in `predict`): 79.3822% (1,028/1,295), no gain. Split identical to `runs/v3_weight_decay_001`. Checkpoint SHA-256 `76e98f870a1d1583e270d95dd394892e5abcae5ea40ae9e1d5e0c7127ef124b4` (weights in ignored `runs/v3_wd_01/model.pth`, not in Git).

Conclusion: no improvement; keep `saved_models/v3_wd001_8093/`.
