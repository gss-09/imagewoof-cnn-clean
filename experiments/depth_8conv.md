# Experiment 2: add one convolution block

Status: completed all 50 epochs on MPS on November 7, 2026. Output: `runs/depth_8conv`. Best validation accuracy: **78.15% at epoch 45**.

## Purpose and design

Test whether adding a fourth block to our six-convolution baseline improves validation accuracy. This follows the assignment's advice to add depth gradually, use pairs of 3×3 convolutions before pooling, and downsample sufficiently for 128×128 inputs.

This is an extension of our existing model. The original three blocks remain structurally identical. The additional block contains two 3×3 Conv → BatchNorm → ReLU sequences with 192 channels, then 2×2 max pooling. All four blocks use max pooling, consistent with our baseline. Global average pooling, dropout 0.3, and a ten-output linear layer finish the model.

| Setting | Completed baseline | Prepared experiment |
| --- | --- | --- |
| Architecture ID | DogCNN-v1 | DogCNN-v2 |
| Convolution layers | 6 | 8 |
| Block channels | 32, 64, 128 | 32, 64, 128, 192 |
| Spatial sizes after pooling | 64, 32, 16 | 64, 32, 16, 8 |
| Trainable parameters | 288,746 | 843,114 |
| Checkpoint / intended output | runs/baseline_full/model.pth | runs/depth_8conv/model.pth |

Adding the block also changes classifier input width and parameter count. This experiment measures the combined effect of that architecture change, not depth independently of capacity.

## Fixed settings

- Same supplied split: 9,065 training and 1,295 validation images; original files remain untouched.
- Seed 42; batch size 32; zero data-loader workers; four CPU threads.
- MPS required for training.
- At most 50 epochs; early-stopping patience 10; best checkpoint selected by validation accuracy.
- AdamW: initial learning rate 0.001, weight decay 0.0001; cosine schedule.
- Label smoothing 0.1; dropout 0.3.
- Identical normalization and augmentation: random resized crop scale 0.7–1.0, horizontal flips, color jitter 0.15. No random augmentation in validation.
- Identical single-view inference for the main comparison; no flip averaging added to the submitted prediction path.

## Checks completed without training

- Both architectures produce finite logits shaped `(N, 10)` from `(N, 3, 128, 128)` inputs.
- Pooling sizes and parameter counts checked on CPU.
- Temporary untrained checkpoint saved and reloaded for both versions; raw-pixel predictions agree before and after reloading.
- The v2 temporary checkpoint passed `verify_pipeline.py`, including isolated fresh-process CPU inference, preprocessing, official labels, input validation, and batch behavior. This checks software behavior, not learned accuracy. The temporary checkpoint was removed automatically.
- The actual trained v1 baseline checkpoint passed the same verifier after the code change.
- Initial preparation used CPU inference only. After authorization, a two-epoch MPS smoke run passed gradient updates, validation, checkpoint reloading, and public prediction verification in `runs/smoke_depth_8conv` (2.2s and 0.8s per epoch; 10% smoke validation accuracy). Full training then started with a fresh seed and model in `runs/depth_8conv`.

## Authorized launch command

The following command was launched from the project root after user authorization. Do not start a duplicate run:

```bash
caffeinate -i python3 -u solution.py --architecture DogCNN-v2 --device mps --output runs/depth_8conv --epochs 50 --batch-size 32 --lr 0.001 --weight-decay 0.0001 --patience 10 --seed 42 --workers 0 --threads 4
```

MPS requires the approved execution environment or a normal macOS Terminal. The output directory must not already exist; it is created only when training starts. Our existing baseline checkpoint remains loadable through `load_model`, which reads the architecture ID from each checkpoint.

## Evaluation after training

1. Confirm the recorded split matches `runs/baseline_full/split.json`.
2. Reload the best checkpoint and verify complete validation accuracy through `predict`; the training script performs this automatically.
3. Run `python3 verify_pipeline.py runs/depth_8conv/model.pth`.
4. Compare best validation accuracy against 70.50%, plus per-breed accuracy, confusion patterns, training curves, best epoch, and elapsed time. Pay attention to English foxhound, golden retriever, and Australian terrier.
5. If evaluating horizontal-flip averaging separately, apply it to both checkpoints and report both single-view and averaged results explicitly.

## Completed results

- Best validation: **78.1467% (1,012/1,295), epoch 45**, versus baseline **70.5019% (913/1,295)**: +7.6448 percentage points, 99 additional correct images.
- All 50 epochs completed; total epoch time **34.77 minutes**. Final epoch validation: 77.37%; training: 85.86% under augmentation.
- Saved split matches the baseline byte for byte. Full validation through the reloaded public `predict` interface exactly reproduced the best score.
- `python3 verify_pipeline.py runs/depth_8conv/model.pth` passed, including isolated CPU loading, labels, preprocessing, repeatability, input validation, and inference batches.
- Golden retriever improved from 57.04% to 74.81%; Australian terrier from 64.44% to 79.26%; English foxhound from 32.50% to 41.25%. English foxhound remains the weakest class. Dingo fell from 78.52% to 75.56%.
- Validation peaked at epoch 45 and fluctuated around 77% afterward while training accuracy stayed around 86%. This suggests a late plateau with some overfitting, rather than evidence that simply appending epochs will help. Augmented training and deterministic validation accuracies are not directly equivalent.
- The cosine learning rate approached zero by epoch 50. A continuation would need an explicit learning-rate plan and a separate output directory; the user subsequently authorized a separate continuation; see [its record](continuation.md). Preserve the best checkpoint and choose any next experiment manually.

