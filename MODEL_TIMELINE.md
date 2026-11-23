# Imagewoof model development timeline

The first Git commit on November 7, 2026, captured the project through DogCNN-v2 at **78.92%**. The final Group 20 package contains DogCNN-v4 at **89.52% (94 of 105)** on a seeded 99/1 re-split. The best result on the original course validation set was **86.72% (1,123 of 1,295)**. The score depends on the validation split, so comparisons below identify the images used.

## November 7 · From the first snapshot to ten convolutions

The [initial snapshot](https://github.com/gss-09/imagewoof-cnn/commit/f2c3331e704b46f3c512c2b86aaf23ca62271aef) records the first three model milestones. The baseline and first v2 runs had already finished when this commit was made. All three used the original course split of 9,065 training and 1,295 validation images.

| Experiment | Change | Top-1 accuracy | Best epoch |
| --- | --- | ---: | ---: |
| [DogCNN-v1 baseline](runs/baseline_full/summary.json) | Three blocks of two 3 × 3 convolutions, with 32/64/128 channels and 288,746 parameters. AdamW, 50 epochs, random crops covering 70–100% of image area. | 70.50% (913/1,295) | 46 |
| [DogCNN-v2](runs/depth_8conv/summary.json) | Added a fourth block at 192 channels, taking the model to eight convolutions and 843,114 parameters. | 78.15% (1,012/1,295) | 45 |
| [v2 continuation](experiments/continuation.md) | Restarted from the v2 weights with AdamW at 0.0001 and cosine decay. Early stopping ended the run after seven extra epochs. | 78.15% retained | 0 extra |
| [v2 gentler crops](experiments/gentler_crop.md) | Raised the minimum crop area from 70% to 85% to preserve more of the dog in each training view. | 78.92% (1,022/1,295) | 47 |
| [v2 random erasing](experiments/random_erasing.md) | Added random erasing with probability 0.25 to the gentler-crop recipe. The gentler-crop checkpoint stayed ahead by nine images. | 78.22% (1,013/1,295) | 41 |
| [DogCNN-v3](experiments/depth_10conv.md) | Added a fifth block and used 32/64/128/256/256 channels, keeping the gentler crops and omitting erasing. | 80.00% (1,036/1,295) | 45 |

The [v3 weight-decay sweep](experiments/v3_weight_decay_001/README.md) increased AdamW weight decay from 0.0001 to 0.001. The resulting checkpoint reached **80.93% (1,048/1,295)** at epoch 45, a gain of 12 correct predictions. [Ten more epochs with zero weight decay](experiments/v3_no_decay_continue/README.md) matched the score and retained the starting weights. Higher decay values of [0.01](experiments/v3_wd_01/README.md) and [0.05](experiments/v3_wd_05/README.md) reached 79.46% and 79.92%, respectively. Training accuracy was about 98% while validation remained near 81%, which motivated changes to the crop range and schedule.

## November 7–8 · Stronger crops, a longer schedule, and wider blocks

These runs kept the original course split. Exponential moving average (EMA) with decay 0.999 smoothed the weights during training, and the later runs saved the EMA checkpoint selected by single-view validation accuracy.

| Round | What changed | Result |
| --- | --- | --- |
| [1](experiments/round1/README.md) | Repeated the v3 control, tested a wider v4 for 50 epochs, and trained v3 for 150 epochs with crops covering 35–100% of image area. | Control 80.77%, wide v4 81.39%, long v3 **82.70% (1,071/1,295)** at epoch 71. The longer schedule and stronger crops were tested together. |
| [2](experiments/round2/README.md) | Combined the 150-epoch crop recipe with DogCNN-v4. The five blocks used 64/128/256/512/512 channels and totaled 9,413,066 parameters. AdamW used learning rate 0.001 and weight decay 0.001. | **83.47% (1,081/1,295)** at epoch 83. |
| [3](experiments/round3/README.md) | Tested 25% minimum crop area with ±10° rotation, then ran the round 2 recipe with seed 43. | Stronger augmentation scored 82.63%. Seed 43 scored 82.93%. The earlier crop range and seed 42 were retained. |

## November 8 · Finding the final training recipe

| Round | What changed | Result on the original course validation set |
| --- | --- | --- |
| [4](experiments/round4/README.md) | Switched to SGD with Nesterov momentum 0.9, weight decay 0.0005, five warm-up epochs, and cosine decay. Compared learning rates 0.05 and 0.1. | **86.10% (1,115/1,295)** at epoch 89 with 0.05, versus 85.64% with 0.1. The 0.05 run gained 34 correct predictions over round 2. |
| [5](experiments/round5/README.md) | Tested a 100-epoch schedule and, separately, SGD weight decay of 0.001 on the 150-epoch schedule. | The shorter run reached 84.63%. Weight decay 0.001 gave **86.72% (1,123/1,295)** through CPU inference at epoch 123. |
| [6](experiments/round6/README.md) | Tested weight decay 0.002 and trained a separate model on all supplied images through epoch 125. | Weight decay 0.002 gave 85.64%. The all-data run had zero held-out images, so it remained unscored. |

The [error analysis](experiments/top2_diagnosis.txt) on the 86.72% checkpoint found 93.13% top-2 accuracy. Beagle and English foxhound accounted for 28 of 172 errors between them. A prediction adjustment toward foxhound lowered overall top-1 accuracy. Flip averaging also lowered the score of this checkpoint, and the public `predict` function kept single-view inference.

## November 8 · Changing the validation split

With the round 5 recipe fixed, the supplied train and validation images were pooled and shuffled within each breed with seed 42. The recorded split files identify the exact training and held-out examples.

| Round | Split | Train / held out | Held-out top-1 | Selected epoch |
| --- | --- | ---: | ---: | ---: |
| [7](experiments/round7/README.md) | 70/30 | 7,251 / 3,109 | 83.92% (2,609/3,109) | 102 |
| [8](experiments/round8/README.md) | 90/10 | 9,324 / 1,036 | 85.91% (890/1,036) | 83 |
| [9](experiments/round9/README.md) | 100/0 | 10,360 / 0 | Unscored | 150, last epoch |
| [10](experiments/round10/README.md) | 95/5 | 9,842 / 518 | 87.84% (455/518) | 126 |
| [11](experiments/round11/README.md) | 95/5, 200-epoch schedule | 9,842 / 518 | 87.45% (453/518) | 162 |
| [11](experiments/round11/README.md) | 100/0, 200-epoch schedule | 10,360 / 0 | Unscored | 200, last epoch |
| [12](experiments/round12/README.md) | 99/1 | 10,255 / 105 | **89.52% (94/105)** | **113** |

On the same 95/5 split, extending the schedule from 150 to 200 epochs cost two correct predictions, so the 150-epoch schedule stayed. The 99/1 split preserved a held-out set while bringing the training set to 10,255 images. Each example on that holdout is about 0.95 percentage points. In the [shared-holdout comparison](experiments/all_models_heldout/results.txt), the 99/1 model got 94 of 105 right and the 90/10 model got 90 of those same images right. Those 105 images also determined the 99/1 checkpoint epoch. The comparison therefore lacks an independent final test set.

## November 8 · Final checkpoint, reproduction, and packaging

The package selected the [round 12 DogCNN-v4 checkpoint](saved_models/v4_resplit9901_8952/README.md). Its 150-epoch training schedule used seed 42 and batch size 32. SGD used learning rate 0.05, Nesterov momentum 0.9, weight decay 0.001, five linear warm-up epochs, and cosine decay. Training views used 35–100% crops resized to 128 × 128, horizontal flips, and color jitter of 0.15. The model used 0.3 dropout, 0.1 label smoothing, and EMA decay 0.999. The checkpoint contains the epoch 113 EMA weights.

Rounds 13 and 14 were excluded from model selection because they used images outside the supplied train and validation pool. See the [experiment selection record](experiments/RECIPE_STORY.md#8-checking-the-choice).

The [round 15 reproduction](experiments/round15/README.md) used the cleaned package source and matched the original split, selected epoch, and **89.52%** score. All 62 tensor storage records matched byte for byte. The saved files have different hashes because the reproduction saves fewer metadata fields. The packaged checkpoint SHA-256 is `2d8f6073d4ae042d9ed7a96c75427dd7d369a5efa90f19471738ef8e0b73f6f9`.

The [Group 20 package record](submission/README.md) documents the two files in the archive, `solution.py` and `model.pth`. The checks loaded the model in evaluation mode, confirmed the ten class IDs against the course mapping, and reproduced 94 of 105 correct predictions through the public interface. They also checked deterministic output, different batch sizes, and single-image and empty batches.

## November 22 · Publishing the result

The [README revision](https://github.com/gss-09/imagewoof-cnn/commit/98c6d12c63e138f636861cd424d4709b789bf079) put the final 99/1 score, the 90/10 and 95/5 results, and the original course validation result together with their denominators. The 80.93% figure belongs to the earlier DogCNN-v3 checkpoint.
