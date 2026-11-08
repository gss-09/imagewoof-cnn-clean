# Round 2: wide network with the long schedule and strong crops

Launched November 7, 2026 (source SHA-256 28e4a25b…, which adds the 2-image tolerance to the final check; training behaviour identical to round 1).

Settings: DogCNN-v4 (64/128/256/512/512, 9,413,066 params), 150 epochs, patience 150, RandomResizedCrop scale 0.35–1.0, AdamW lr 0.001 wd 0.001, cosine, batch 32, label smoothing 0.1, dropout 0.3, EMA 0.999 (saved), seed 42, supplied split, workers 6.

## Result: new best

| Run | Raw best | EMA best (saved) | Public `predict` | Flip-averaged (saved checkpoint, separate script) | Final train |
| --- | --- | --- | --- | --- | --- |
| `r2_v4_long150` | 83.40% @139 | **83.47% @83** | 83.47% (1,081/1,295) | 84.40% (1,093) | 99.7% |

- +33 images over the 80.93% reference, +10 over `r1_v3_long150`. CPU verifier passed. Best per-epoch flip score 84.71% @102.
- Per class (single view): shih_tzu 85.9, rhodesian_ridgeback 86.7, beagle 82.2, english_foxhound 56.2, border_terrier 83.7, australian_terrier 81.5, golden_retriever 80.0, old_english_sheepdog 89.6, samoyed 95.6, dingo 82.2.
- Bundled as `saved_models/v4_long150_8347/` (weights tracked in Git), tag `v4-long150-83.47`. Checkpoint SHA-256 3303616c…
- Still memorizes (99.7% train): stronger regularization/augmentation is the most promising next lever. A second seed would measure how much of the gain is luck.
