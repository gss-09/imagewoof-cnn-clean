# DogCNN-v4, SGD lr 0.05, 150 epochs: 86.10% validation

Reached a new best on the original course split on November 8, 2026. From `runs/r4_v4_sgd_lr005`, seed 42, supplied split.

- Architecture DogCNN-v4: 10 conv 3x3 (64/128/256/512/512), BN+ReLU, max pool after each pair, GAP, dropout 0.3, linear; 9,413,066 parameters; sequential, no skips.
- SGD Nesterov momentum 0.9, lr 0.05, weight decay 0.0005, 5 linear warm-up epochs then cosine to epoch 150, batch 32, label smoothing 0.1, RandomResizedCrop scale (0.35, 1.0), horizontal flip, ColorJitter 0.15.
- Saved weights are the exponential moving average (decay 0.999) of the training weights, selected at epoch 89.
- Validation: 86.1004% (1,115/1,295) through public `predict` (single view). CPU verifier passed.
- Flip averaging (not in `predict`): 86.4093% (1,119/1,295), measured separately.

Load: `from solution import load_model, predict; m = load_model("model.pth")`.
