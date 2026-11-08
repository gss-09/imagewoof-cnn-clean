# DogCNN-v3, AdamW weight decay 0.05

Hypothesis: raising weight decay 0.0001 -> 0.001 gave +12 images (80.93%) while train accuracy still reached ~98%; stronger decoupled decay (0.05) may reduce overfitting further. Only change versus `runs/v3_weight_decay_001`: weight decay 0.05. Fresh 50-epoch run, seed 42, supplied split. Log: `setup/v3_wd_05.log`.

Status: launched and completed November 7, 2026.

## Result

Completed all 50 epochs. Best epoch 43: **79.9228% (1,035/1,295)**, reproduced through public `predict` after reload; 13 images fewer than the 0.001 run (80.93%). Final epoch 50: train 98.75%, validation 79.92%, so stronger decay did not prevent overfitting by the end. Flip averaging, measured separately on CPU with a scratch script (not in `predict`): 81.0039% (1,049/1,295), still below the 0.001 run's 81.8533% flip-averaged. Split identical to `runs/v3_weight_decay_001`. Checkpoint SHA-256 `67af6b5c639b5b62f21d9188b98fd6c623deb66b5ec8dcb6f36e68cda4692e10` (weights in ignored `runs/v3_wd_05/model.pth`, not in Git).

Conclusion: no improvement; keep `saved_models/v3_wd001_8093/`.
