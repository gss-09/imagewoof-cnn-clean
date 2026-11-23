# DogCNN-v4, 150 epochs, crop 0.35: 83.47% validation

Reached a new best on the original course split on November 7, 2026. From `runs/r2_v4_long150`, seed 42, supplied split.

- Architecture DogCNN-v4: 10 conv 3x3 (64/128/256/512/512), BN+ReLU, max pool after each pair, GAP, dropout 0.3, linear; 9,413,066 parameters; sequential, no skips.
- AdamW lr 0.001, wd 0.001, cosine over 150 epochs, batch 32, label smoothing 0.1, RandomResizedCrop scale (0.35, 1.0), horizontal flip, ColorJitter 0.15.
- Saved weights are the exponential moving average (decay 0.999) of the training weights, selected at epoch 83.
- Validation: 83.4749% (1,081/1,295) through public `predict` (single view). CPU verifier passed.
- Flip averaging (not in `predict`): 84.4015% (1,093/1,295), measured separately.

Load: `from solution import load_model, predict; m = load_model("model.pth")`.
