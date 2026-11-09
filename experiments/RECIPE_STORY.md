# How we got to the final recipe (COSC 4368 HW2, Nov 7–8, 2026)

Every number below is validation accuracy measured through the public `predict` function, single view, seed 42. Each experiment changed one thing at a time, chosen by hand (no tuning libraries). Details for each step are in `experiments/` and `runs/`.

## 1. A small, safe baseline: 70.50%

We started with what the assignment suggests: pairs of 3×3 convolutions with BatchNorm and ReLU, a max-pool after each pair, global average pooling, dropout and one dense layer.

- **DogCNN-v1**: 6 convolutions (32/64/128 channels), 288,746 parameters, AdamW lr 0.001, 50 epochs, crops covering 70–100% of the image, horizontal flip, small color jitter, label smoothing 0.1.
- Result: **70.50%** (913/1,295) at epoch 46, trained on the Mac (MPS).

We also checked the label order carefully: folders are read in `classes.txt` order, never alphabetical order.

## 2. Going deeper: 78.15% → 80.00%

The baseline was too small to tell 10 similar breeds apart, so we added depth.

| Step | Change | Result |
| --- | --- | --- |
| DogCNN-v2 | 4th block (192 ch), 8 convolutions, 843k params | **78.15%** (+7.6) |
| Fine-tune v2 10 more epochs | lower learning rate | no gain |
| Gentler crops | crop 85–100% instead of 70–100% | 78.92% |
| Random erasing | blank out random patches | 78.22% (hurt) |
| DogCNN-v3 | 5 blocks, 10 convolutions (32/64/128/256/256) | **80.00%** |

## 3. Regularization: 80.93%

Training accuracy hit ~98% while validation stalled near 80%: the network was memorizing.

- Weight decay 0.001 (from 0.0001): **80.93%**. Values 0.01 and 0.05 were worse (79.46%, 79.92%).
- A control rerun reproduced 80.93% within one image, so our noise level is about ±0.5–1 point. Smaller differences than that are luck.
- Using 6 data-loading workers cut epoch time from ~21 s to ~8 s.

## 4. The big lesson: stronger crops + longer training: 82.70% → 83.47%

| Run | Change | Result |
| --- | --- | --- |
| v3, 150 epochs, crops 35–100% | stronger augmentation, longer schedule | **82.70%** (+1.8) |
| v4: every block twice as wide (64/128/256/512/512, 9.4M params) | more capacity, 50 epochs | 81.39% |
| v4 + 150 epochs + crops 35–100% + EMA 0.999 | combine both | **83.47%** |

EMA (exponential moving average) keeps a smoothed running average of the weights during training; we evaluate and save that averaged copy.

What did not help (round 3): even stronger augmentation (crops 25–100% + rotation ±10°) 82.63%, and a different seed 82.93% (more evidence of ~0.5-point noise).

## 5. Switching the optimizer: 86.10% → 86.72%

| Run | Change | Result |
| --- | --- | --- |
| SGD with Nesterov momentum 0.9, lr 0.05, 5 warm-up epochs + cosine decay | replaced AdamW | **86.10%** (+2.6) |
| Same, lr 0.1 | | 85.64% |
| Same, weight decay 0.001 | | **86.72%** (1,123/1,295) |
| Same, 100 epochs | shorter schedule | 84.63% |
| Same, weight decay 0.002 | | 85.64% |

This was the final recipe. Everything after this kept it fixed and only changed how much data the model trained on.

## 6. Understanding the errors

On the 86.72% model:

- Top-2 accuracy was 93.13%: when wrong, the right breed was usually its second guess.
- The biggest confusion was beagle ↔ English foxhound (28 of 172 errors). Foxhound also has the fewest images.
- Nudging predictions toward foxhound fixed some foxhounds but cost more elsewhere (86.72% → 86.56%), so we dropped it.
- Averaging each image with its mirror image at prediction time ("flip averaging") did not reliably help, so `predict` uses a single view.

## 7. More training data: re-splitting train + validation

The instructor allowed pooling the supplied train and validation images and re-splitting them. We re-split each breed with seed 42 and kept the same recipe:

| Split (train/validation) | Training images | Result |
| --- | --- | --- |
| 70/30 | 7,251 | 83.92% (on 3,109) |
| 90/10 | 9,324 | 85.91% (on 1,036) |
| supplied split | 9,065 | 86.72% (on 1,295) |
| 95/5 | 9,842 | 87.84% (on 518) |
| **99/1** | **10,255** | **89.52% (94/105), epoch 113** |

More training data helped at every step.

We also tried 200 epochs instead of 150: on the 95/5 split it scored 87.45%, two images worse, and memorized more. Every measured run peaked between epochs 83 and 126, so 150 stays.

A 100/0 version (all 10,360 images) has no validation images, so its stopping epoch would be a guess and its accuracy can't be measured. We chose 99/1: almost all the data, and the epoch is still chosen by validation accuracy.

## 8. Checking the choice

- We scored every saved model only on validation images it never trained on. On the 105 images that neither model trained on, the 99/1 model got 94 right and the 90/10 model got 90 (`experiments/all_models_heldout/`).
- Two runs (r13, r14) that we did not make were selected on an outside copy of what is likely the hidden test set. They are excluded and must not be used.

## 9. The final submission

- **Model:** `saved_models/v4_resplit9901_8952/model.pth` (SHA-256 2d8f6073…).
- **Code:** a cleaned `solution.py` (`submission/staged/`) with only the code that trained this model; its defaults are the final recipe. Replaying it matched the original run's log digit for digit; a full retrain to confirm (round 15) was in progress at the time of writing.

### Final recipe

- **Data:** supplied train + validation pooled (10,360), re-split 99/1 per breed with seed 42 → 10,255 train / 105 validation. 128×128 RGB, pixels scaled to [-1, 1].
- **Model (DogCNN-v4):** 5 blocks of [3×3 conv → BatchNorm → ReLU] × 2 + max-pool, channels 64/128/256/512/512 → global average pooling → dropout 0.3 → dense 10. 10 convolutions, 9,413,066 parameters, sequential, trained from scratch.
- **Augmentation (training only):** random crop of 35–100% of the area resized to 128×128, horizontal flip, color jitter 0.15.
- **Training:** SGD, Nesterov momentum 0.9, lr 0.05, weight decay 0.001, batch 32, 5 warm-up epochs then cosine decay over 150 epochs, label smoothing 0.1, EMA of weights (decay 0.999).
- **Selection:** EMA weights from the epoch with the highest validation accuracy: epoch 113, 89.52% (94/105).
