# DogCNN-v4, SGD lr 0.05, weight decay 0.001, 150 epochs: 86.72% validation

Best result on the original course validation split. From `runs/r5_v4_sgd_wd001`, seed 42.

- Architecture DogCNN-v4: 10 conv 3x3 (64/128/256/512/512), BN+ReLU, max pool after each pair, GAP, dropout 0.3, linear; 9,413,066 parameters; sequential, no skips.
- SGD Nesterov momentum 0.9, lr 0.05, weight decay 0.001, 5 linear warm-up epochs then cosine to epoch 150, batch 32, label smoothing 0.1, RandomResizedCrop scale (0.35, 1.0), horizontal flip, ColorJitter 0.15.
- Saved weights are the exponential moving average (decay 0.999) of the training weights, selected at epoch 123.
- Validation: 86.7181% (1,123/1,295) single view on CPU (verifier passed); 86.6409% (1,122) through public `predict` on accelerator (one near-tied image, accelerator rounding).
- Flip averaging (not in `predict`): 86.4865% (1,120/1,295), measured separately; it does not help this checkpoint.

Load: `from solution import load_model, predict; m = load_model("model.pth")`.
