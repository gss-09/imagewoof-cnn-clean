# Training process and selected result

## Selected checkpoint

Use `best_model/model.pth` with the matching `best_model/solution.py`. This is an exact copy of `runs/depth_8conv/model.pth`, selected at epoch 45 with **78.1467% validation accuracy (1,012/1,295)**. File hashes are saved in `best_model/manifest.json`. Hidden-test accuracy is unknown.

## Data and architecture

Only supplied images were used: 9,065 training and 1,295 validation images. The supplied split stayed unchanged for all three full experiments; run folders retain identical split records. Images are 128×128 RGB. Labels follow the official numeric order in `classes.txt`, rather than alphabetical folder order.

Our custom sequential CNN was initialized from scratch. DogCNN-v2 has four blocks with channels 32, 64, 128, 192. Each block contains two 3×3 convolutions with BatchNorm and ReLU, followed by 2×2 max pooling. Global average pooling, dropout 0.3, and a ten-output linear layer finish the network. There are eight convolutions and 843,114 trainable parameters. No pretrained weights, skip connections, attention, external images, or automated hyperparameter search were used. This extends our own baseline.

Completed-run training augmentation: random resized crop to 128×128 with scale 0.7–1.0, random horizontal flip, and brightness/contrast/saturation jitter 0.15. Both training and validation convert pixels to [0,1] and normalize with mean/std 0.5 per channel. Validation is deterministic, with no random augmentation. Public prediction uses a single view, without flip averaging.

## Manual experiments

| Experiment | Settings | Result |
| --- | --- | --- |
| Baseline, `runs/baseline_full` | Six convolutions, channels 32/64/128; 288,746 parameters; 50 epochs | 70.5019% (913/1,295), best epoch 46; 35.15 minutes |
| Added block, `runs/depth_8conv` | Eight convolutions, channels 32/64/128/192; 843,114 parameters; 50 epochs | **78.1467% (1,012/1,295)**, best epoch 45; 34.77 minutes |
| Continuation, `runs/depth_8conv_continue` | Start from v2 epoch 45; fresh AdamW at 0.0001; cosine over at most 20 extra epochs; patience seven | Stopped after seven extra epochs, 5.02 minutes. Best extra epoch 77.22%; original 78.1467% retained |

The first two runs used AdamW with initial learning rate 0.001, weight decay 0.0001, cosine decay, cross-entropy with label smoothing 0.1, batch size 32, seed 42, early-stopping patience ten, zero loader workers, and four CPU threads. Training ran on the Mac's MPS accelerator. Fixed seeds were recorded; deterministic algorithms warn on unsupported operations, so results across hardware are not guaranteed identical.

Adding the fourth block improved validation by 7.6448 percentage points, or 99 images, on the same split. Depth and capacity changed together, so the result does not isolate depth alone. The continuation restarted the optimizer and schedule because checkpoints did not contain optimizer state. It never improved validation, so the original best weights remain selected. Improving training performance without validation gains suggests a plateau with some overfitting under these settings; it does not prove that every other training choice would fail.

Exact commands and interpretation are in [the v2 report](depth_8conv.md) and [the continuation report](continuation.md). Original run folders retain configuration, history, summary, and split records. The baseline also retains plots, a confusion matrix, and its results report.

## Verification and cleanup

The best weights reproduced 78.1467% through the public `predict` interface on all 1,295 validation images. CPU verification checks official labels, raw-pixel preprocessing, output shape/type/range, repeatability, single and multiple batches, empty and malformed inputs, and loading in an isolated fresh process.

Five disposable smoke-run folders were removed after preserving their complete JSON records in `experiments/smoke_checks.json`. Their subset scores were software checks, not full model results. The continuation's redundant weights were removed only after confirming every model tensor matched the source; `runs/depth_8conv_continue/checkpoint_reference.json` points to the retained checkpoint. Python bytecode cache and root Finder metadata were removed. Full experiment records, both original trained checkpoints, source data and course materials were preserved.

To check the saved model bundle from the project root:

```bash
python3 verify_pipeline.py best_model/model.pth
```

No training remains active. The model bundle has not been submitted; group information and the contribution report still need completion before packaging the assignment.

## Gentler crop completed

Root `solution.py` now raises the minimum crop area from 0.7 to 0.85. The saved best bundle retains its original code and weights. See [gentler crop experiment](gentler_crop.md); the fresh v2 CUDA run completed 50 epochs in `runs/gentler_crop_085`. Best epoch 47 reached 78.9189% (1,022/1,295), reproduced through public prediction. This was the highest measured result before v3; the historical `best_model/` bundle remains intact. Full run records, weights, and exact training source have been copied to the laptop.

## Next experiment and version history

Random erasing completed 50 epochs: best 78.2239% (1,013/1,295), epoch 41; public prediction and CPU verification passed. It trails gentler crop by 9 correct images. Keep the 78.9189% checkpoint. No training remains active; see [settings](random_erasing.md). Local Git tag `v2-crop085-78.92` preserves the completed CUDA model code. `model_registry.json` records checkpoint hashes. Weights remain outside Git in local run folders.

## Larger architecture preparation (historical)

Before the authorized run, DogCNN-v3 was implemented without training; see [plan](depth_10conv.md). Best 78.92% model and exact Python source are archived in `saved_models/v2_crop085_7892/` and committed including weights under tag `best-v2-7892-bundle`. Historical bundles and original runs remain intact.

## Completed larger architecture

DogCNN-v3 completed 50 CUDA epochs: best epoch45 80.00% (1,036/1,295), 14 more correct images than v2 gentler crop. Public prediction and CPU portability checks passed. Exact source and weights are saved with verified hashes. New best bundle: `saved_models/v3_depth10_8000/`, intentionally committed including weights under tag `v3-depth10-80.00`. Earlier bundles remain intact. No training remains active. See [results](depth_10conv.md).
