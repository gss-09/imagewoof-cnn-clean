# Round 8: 86.72% recipe on a seeded, stratified 90/10 re-split

Launched November 8, 2026, source SHA-256 ad63459a… (`--resplit 0.1`, seed 42).
Pooled the supplied 10,360 train+val images; per breed, 10% held out → 9,324 train / 1,036 validation.

| Metric | Value |
| --- | --- |
| EMA best (saved), epoch 83 | **85.91% (890/1,036)**, reproduced by public `predict` |
| Final epoch 150 | 84.75% EMA, train 99.64% |
| Flip-averaged best (logged only) | 86.78% @149 |
| Checkpoint SHA-256 | 1a034107… (verified) |

Per breed at the saved epoch: shih_tzu 86.1, rhodesian_ridgeback 87.0, beagle 89.8, english_foxhound 70.3, border_terrier 92.6, australian_terrier 82.4, golden_retriever 76.9, old_english_sheepdog 84.3, samoyed 93.5, dingo 89.8.

Findings:
- Not directly comparable with 86.72% (1,295 supplied val images); differences under ~1 point are within noise (1 image ≈ 0.1 point here).
- Split trend: 70/30 83.92% → 90/10 85.91% → supplied 86.72%. More training data helps, but the gain from +3% data is too small to see.
- Peak came earlier (epoch 83) than on the supplied split (123); weakest breeds remain English foxhound and golden retriever.
- No new best; `saved_models/v4_sgd_wd001_8672/` remains the measured best.
