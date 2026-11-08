# Round 11: 200-epoch schedule on 100/0 and 95/5

Launched November 8, 2026, source SHA-256 ad63459a…, `--epochs 200 --patience 200`, otherwise the 86.72% recipe.

- `r11_v4_trainval_ep200` (100/0, `--train-on-val`): finished 200 epochs, final train 99.7%; unscored (no held-out images); CPU verifier passed. Checkpoint SHA-256 71317948….
- `r11_v4_resplit9505_ep200` (same 95/5 split as round 10): EMA best epoch 162 **87.45% (453/518)**, public `predict` 87.45%; final epoch ~85.3%. Checkpoint SHA-256 30e6b7eb… (verified). Best flip-averaged per-epoch score 88.61% @167 (logged only).
- Conclusion: 200 epochs is 2 images worse than 150 epochs (87.84%) on the same split and memorizes harder. Keep the 150-epoch schedule; the 100/0 ep200 model is not preferred over ep125.
