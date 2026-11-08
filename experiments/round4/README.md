# Round 4: SGD instead of AdamW on the best recipe

November 8, 2026. Launched; source `solution.py` SHA-256 28607dec… (unchanged; `--optimizer sgd` already existed).

Everything as `r2_v4_long150` (DogCNN-v4, 150 epochs, batch 32, crop 0.35–1.0, flip, ColorJitter, label smoothing 0.1, dropout 0.3, EMA 0.999 saved, seed 42, supplied split) except the optimizer: SGD, Nesterov momentum 0.9, weight decay 0.0005, 5 linear warm-up epochs then cosine.

| Run | lr | EMA best (saved) | Public `predict` | Flip-averaged (separate script) | Raw best | Final train |
| --- | --- | --- | --- | --- | --- | --- |
| `r2_v4_long150` (reference, AdamW) | 0.001 | 83.47% @83 | 83.47% (1,081) | 84.40% (1,093) | — | — |
| `r4_v4_sgd_lr005` | 0.05 | **86.10% @89** | 86.10% (1,115) | 86.41% (1,119) | 85.10% @150 | 99.8% |
| `r4_v4_sgd_lr01` | 0.1 | 85.64% @118 | 85.64% (1,109) | 85.71% (1,110) | 85.33% @146 | 99.7% |

Findings:

- SGD beat AdamW by +2.6 points (lr 0.05) and +2.2 (lr 0.1), both well beyond the ~0.5-point seed noise from round 3.
- lr 0.05 vs 0.1 differ by 6 images, within noise.
- lr 0.05 peaked at epoch 89 and the EMA drifted down ~1.5 points as train accuracy reached ~100%; best-checkpoint selection kept the peak.
- English foxhound remains weakest (70.0% at lr 0.05, up from 56.25% in the AdamW best).
- New best bundle: `saved_models/v4_sgd005_8610/` (CPU verifier passed), tag `v4-sgd005-86.10`.
