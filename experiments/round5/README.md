# Round 5: shorter schedule vs stronger weight decay (SGD recipe)

Launched November 8, 2026 (source SHA-256 28607dec…). Base: `r4_v4_sgd_lr005` (86.10%): DogCNN-v4, SGD Nesterov 0.9, lr 0.05, wd 0.0005, 5 warm-up + cosine, crop 0.35, batch 32, EMA 0.999, seed 42, supplied split. One change per run.

| Run | accelerator | Change | EMA best (saved) | Public `predict` | Flip (separate) | Final train |
| --- | --- | --- | --- | --- | --- | --- |
| `r5_v4_sgd_ep100` | 0 | 100 epochs | 84.63% @53 (1,096) | 84.63% | — | 99.3% |
| `r5_v4_sgd_wd001` | 1 | wd 0.001 | **86.72% @123 (1,123)** | 86.64% accelerator (1,122) / 86.72% CPU | 86.49% (1,120) | 99.7% |

Findings:

- The shorter schedule memorized sooner (96% train by epoch 70) and peaked early at 84.63%: −1.5 points. Keep 150 epochs.
- Doubling SGD weight decay to 0.001 slowed memorization (train 85% at epoch 84 vs ~96%+ before) and pushed the peak later (epoch 123), giving a new best +0.62 point (+8 images). Seed noise is ~±0.5, so the gain is modest.
- accelerator vs CPU public predict differ by one near-tied image (within the 2-image tolerance).
- Flip averaging slightly hurts this checkpoint (−3 images); English foxhound fell to 56% (70% in the 86.10% run).
- New best bundle: `saved_models/v4_sgd_wd001_8672/`, tag `v4-sgdwd001-86.72`. CPU verifier passed on the bundle.
