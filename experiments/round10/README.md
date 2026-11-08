# Round 10: 86.72% recipe on a seeded, stratified 95/5 re-split

Launched November 8, 2026, source SHA-256 ad63459a…, `--resplit 0.05`, seed 42 → 9,842 train / 518 validation.

- EMA best (saved) epoch 126: **87.84% (455/518)**, reproduced by public `predict`. Checkpoint SHA-256 8e71ad81… (verified).
- Per breed: shih_tzu 81.5, rhodesian_ridgeback 90.7, beagle 87.0, english_foxhound 68.8 (≈26 images), border_terrier 94.4, australian_terrier 85.2, golden_retriever 87.0, old_english_sheepdog 94.4, samoyed 94.4, dingo 87.0.
- Split trend: 70/30 83.92% → 90/10 85.91% → supplied 86.72% → 95/5 87.84%. Consistent with "more training data helps", but 518 images give ±~1.5 points noise and a different validation set; not proof of a better model.
- Peak epoch 126 (supplied split 123) supports stopping near 125 for the 100/0 retrain (`runs/r6_v4_trainval_ep125`).
