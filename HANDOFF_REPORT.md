# CNN assignment: work report and agent handoff

Prepared after the evening experiment session, November 7, 2026. This report covers the accumulated work in this conversation, including the earlier baseline, so a new agent can continue without the chat history.

## 1. Current state: read this first

- **Update November 8, 2026 (round 5): new best 86.7181% (1,123/1,295)**, the SGD recipe with weight decay 0.001, EMA best epoch 123. Bundle **`saved_models/v4_sgd_wd001_8672/`**, tag **`v4-sgdwd001-86.72`**. Flip 86.49% (no help). 100-epoch schedule gave 84.63%. See `experiments/round5/README.md`. No training active.
- Earlier November 8 update: 86.1004% (1,115/1,295)**, same DogCNN-v4 recipe but SGD Nesterov (momentum 0.9, lr 0.05, wd 0.0005, 5 warm-up epochs, cosine), EMA best epoch 89. Bundle **`saved_models/v4_sgd005_8610/`**, tag **`v4-sgd005-86.10`**. Flip averaging measured separately 86.41%; `predict` still single view. SGD lr 0.1 gave 85.64%. See `experiments/round4/README.md`. No training active.
- **Update November 7, 2026 (late): new best 83.4749% (1,081/1,295)**, DogCNN-v4, 150 epochs, crops 0.35–1.0, EMA 0.999 weights, best epoch 83. Bundle **`saved_models/v4_long150_8347/`**, tag **`v4-long150-83.47`**. Flip averaging measured separately 84.40%; `predict` still single view. See `experiments/round1/README.md`, `experiments/round2/README.md` and `experiments/round3/README.md` (round 3: no new best; seed 43 gave 82.93%, stronger aug 82.63%). Round runs: `r1_v3_control` 80.77%, `r1_v4_wide` 81.39%, `r1_v3_long150` 82.70%, `r2_v4_long150` 83.47%. No training active. The bullets below describe the earlier state (80.93% best) and are kept for history.

- Best measured validation result: **80.9266409% (1,048/1,295)**, DogCNN-v3, epoch45 of a 50-epoch run with AdamW weight decay0.001.
- Selected bundle: **`saved_models/v3_wd001_8093/`**. Contains the actual trained `model.pth`, exact `solution.py`, requirements, config, summary, and SHA-256 manifest. Binary weights intentionally committed to local Git, tag **`v3-wd001-80.93`**.
- Most recent experiment: continued that checkpoint for10 extra epochs with weight decay0 and starting learning rate0.00001. **No improvement.** Extra epoch10 tied80.9266%; the selected checkpoint remained initialization epoch0, with state tensors exactly identical to the original epoch45 model.
- **No training remains active from our experiments.**
- Latest local Git commit: **`837fa9a`**, recording the unsuccessful continuation. Working tree was clean after that commit, before adding this report.
- No assignment submission or final submission archive has been made. Group information and contribution report remain outstanding.
- User wants meaningful improvements, a compliant independent implementation, saved model versions in local Git, and live monitoring when requested.

## 2. Local environment

Local project:

```text
/Users/shriyanssaiganipisetti/Desktop/Intro to ai assignment 2
```

The Mac's MPS accelerator also works outside the restricted agent sandbox. A sandbox-only false availability result is not evidence that PyTorch needs reinstalling.

## 3. Assignment and data requirements

Read `AGENTS.md`, `HW2-1.pdf`, and `extra directions.txt` before making changes. Lecture references: `lectures/lecture4_MLP (1).pptx`, `lecture5_CNN.pptx`, `lecture6_RNN.pptx`, `lecture7_transformer.pptx`.

Task: classify ten Imagewoof dog breeds using a custom **sequential CNN trained from scratch**. Standard convolution, activation, pooling/GAP, BatchNorm, Dropout, Flatten, Dense layers are permitted. No pretrained models/weights, residual or skip connections, dense blocks, attention, transformers, or automated HPO/NAS tools. Manual tuning is permitted. The suggested4–6 convolutions are a starting suggestion, not a maximum; our10-convolution network remains sequential.

Only supplied images may be used. No external Imagewoof/ImageNet downloads or hidden-test access. Instructor permits combining and re-splitting supplied train/validation data, including cross-validation, but **we have kept the supplied split throughout these comparisons**.

- Training:9,065 images under `imagewoof_train/train/`.
- Validation:1,295 under `imagewoof_val/val/`.
- All supplied10,360 PNGs were verified as RGB128×128 without image errors.
- Hidden test set:2,591 images, unavailable here. All reported results are validation results.
- English foxhound has fewer examples; class-level results matter.

Official labels, from both `classes.txt` files, must retain this order:

| ID | Class | Train | Validation |
| --- | --- | ---: | ---: |
| 0 | shih_tzu | 945 | 135 |
| 1 | rhodesian_ridgeback | 945 | 135 |
| 2 | beagle | 945 | 135 |
| 3 | english_foxhound | 562 | 80 |
| 4 | border_terrier | 944 | 135 |
| 5 | australian_terrier | 944 | 135 |
| 6 | golden_retriever | 945 | 135 |
| 7 | old_english_sheepdog | 945 | 135 |
| 8 | samoyed | 945 | 135 |
| 9 | dingo | 945 | 135 |

Never use alphabetical folder IDs without remapping. Current custom loader explicitly follows official IDs. Validation is deterministic and never receives random training augmentation or gradient updates.

## 4. Implementation built

Root `solution.py` is self-contained and contains training CLI plus grader interface. `verify_pipeline.py` checks data labels, preprocessing, prediction behavior and CPU loading. Importing `solution.py` does not train or download data.

Each architecture block is two3×3, padding1, bias-free convolutions, each followed by BatchNorm and ReLU, then2×2 MaxPool. The classifier is adaptive global average pooling, Flatten, Dropout0.3, Linear to10 classes.

| Architecture | Block channels | Convolutions | Parameters |
| --- | --- | ---: | ---: |
| DogCNN-v1 | 32,64,128 | 6 | 288,746 |
| DogCNN-v2 | 32,64,128,192 | 8 | 843,114 |
| DogCNN-v3 | 32,64,128,256,256 | 10 | 2,356,458 |

V3 spatial sizes after pooling:64,32,16,8,4. No skip connections or pretrained components.

CLI default remains **DogCNN-v1**. Explicitly select `--architecture DogCNN-v3` for v3 experiments; do not accidentally run the default. `train_mps.command` is historical and selects v1. Checkpoint loading selects the architecture recorded in metadata.

Current training augmentation: RandomResizedCrop128, scale0.85–1.0; horizontal flip; ColorJitter0.15; ToTensor; normalization mean/std=(0.5,0.5,0.5). No random erasing in root code. Earlier v1/v2 used crop minimum0.7. Exact historical source is saved per run.

Required public interface:

```python
def load_model(path="model.pth"):
    ...

def predict(model, images):
    ...
```

`load_model` portably loads on CPU with `weights_only=True`, builds the recorded architecture and returns eval mode. `predict` accepts raw RGB uint8 NumPy `(N,128,128,3)`, converts to NCHW, divides by255, normalizes internally, performs inference in batches32 and returns NumPy int64 official IDs `(N,)`. Single, multiple and empty inputs supported; invalid inputs rejected. No training-data dependency for inference. Single-view inference only; horizontal-flip averaging was measured separately historically but is not implemented in this path.

Training saves the strict best validation checkpoint, full history/config/split/summary; after finishing, it reloads the best checkpoint and checks full validation accuracy through public `predict`. `--init-checkpoint` loads weights but creates a **fresh optimizer and fresh cosine schedule**. It evaluates the starting checkpoint, saves it as continuation epoch0, and keeps it unless a later epoch strictly improves. Checkpoints do not resume the old optimizer state.

## 5. Experiment history and results

All full runs used seed42 and the supplied split. Full fresh runs used AdamW, batch32, starting lr0.001, cosine50, label smoothing0.1, Dropout0.3, patience10. Weight decay was0.0001 except the final stronger-decay run. Workers0, threads4. Current continuation used the specific settings below.

| Run directory | Change / hardware | Epochs completed | Best epoch | Best validation |
| --- | --- | ---: | ---: | ---: |
| `runs/baseline_full` | v1, MPS | 50 | 46 | 70.5019%,913/1295 |
| `runs/depth_8conv` | v2, MPS | 50 | 45 | 78.1467%,1012/1295 |
| `runs/depth_8conv_continue` | v2 continuation, lr0.0001, wd0.0001, max20, patience7 | 7 | 0, original retained | 78.1467%, no gain |
| `runs/gentler_crop_085` | v2, crop minimum0.7→0.85, CUDA | 50 | 47 | 78.9189%,1022/1295 |
| `runs/random_erasing_025` | v2, erasing added, CUDA | 50 | 41 | 78.2239%,1013/1295 |
| `runs/depth_10conv` | v3, no erasing, CUDA | 50 | 45 | 80.0000%,1036/1295 |
| `runs/v3_weight_decay_001` | v3, wd0.0001→0.001, CUDA | 50 | 45 | **80.9266%,1048/1295** |
| `runs/v3_no_decay_continue` | best v3 continuation, wd0, lr0.00001, cosine10 | 10 | 0, original retained | 80.9266%, no gain |
| `runs/v3_wd_01` | v3, wd 0.01 | 50 | 39 | 79.4595%, 1029/1295 |
| `runs/v3_wd_05` | v3, wd 0.05 | 50 | 43 | 79.9228%, 1035/1295 |
| `runs/r1_v3_control` | v3 repeat, workers 6, EMA | 50 | 42 (EMA) | 80.7722%, 1046/1295 (public CPU) |
| `runs/r1_v4_wide` | v4 (2x wide), EMA | 50 | 34 (EMA) | 81.3900%, 1054/1295 |
| `runs/r1_v3_long150` | v3, 150 epochs, crop 0.35, EMA | 150 | 71 (EMA) | 82.7027%, 1071/1295 |
| `runs/r2_v4_long150` | v4, 150 epochs, crop 0.35, EMA | 150 | 83 (EMA) | 83.4749%, 1081/1295 |
| `runs/r3_v4_aug_strong` | v4, 150 epochs, crop 0.25 + rotation ±10°, EMA | 150 | 88 (EMA) | 82.6255%, 1070/1295 |
| `runs/r3_v4_seed43` | as r2_v4_long150, seed 43 | 150 | 53 (EMA) | 82.9344%, 1074/1295 |
| `runs/r4_v4_sgd_lr005` | as r2_v4_long150 but SGD Nesterov lr 0.05, wd 0.0005, warm-up 5 | 150 | 89 (EMA) | 86.1004%, 1115/1295 |
| `runs/r4_v4_sgd_lr01` | same, SGD lr 0.1 | 150 | 118 (EMA) | 85.6371%, 1109/1295 |
| `runs/r5_v4_sgd_ep100` | as r4_v4_sgd_lr005, 100 epochs | 100 | 53 (EMA) | 84.6332%, 1096/1295 |
| `runs/r5_v4_sgd_wd001` | as r4_v4_sgd_lr005, wd 0.001 | 150 | 123 (EMA) | **86.7181%, 1123/1295** (CPU; accelerator predict 1122) |

Interpretation:

1. Increasing depth from v1 to v2 produced the largest improvement: +7.6448 percentage points,99 images.
2. Gentler cropping improved v2 by10 images. Random erasing reduced performance by9 images against that reference, so it was removed.
3. V3 improved on best v2 by14 images, to80%.
4. Stronger weight decay improved v3 by12 images, +0.9266 percentage points.
5. Continuation attempts on both v2 and final v3 gave no improvement. Do not describe an epoch0 retained score as newly improved training performance.
6. Best-to-baseline gain:135 more correct images, +10.4247 percentage points.

Random erasing settings were p0.25, scale0.02–0.12, ratio0.5–2.0, value0 after normalization, training only. Its exact source remains in its run directory.

### Final successful v3 run

About21 seconds per epoch, roughly18 minutes total.

Validation milestones: epoch27=77.61%,32=78.61%,33=79.46%,36=79.54%,38=80.62%,44=80.77%,45=80.93%. Final epoch50: train98.18%, validation80.54%. Best weights were reloaded and public prediction reproduced80.9266%.

Learning rate already decayed substantially: epoch45 approximately0.00003511; epoch50 approximately0.000000987. Thus simply adding epochs at the end of the original schedule was not an obvious solution.

Best checkpoint per-class validation:

| Class | Accuracy |
| --- | ---: |
| shih_tzu | 85.93% |
| rhodesian_ridgeback | 84.44% |
| beagle | 76.30% |
| english_foxhound | 61.25% |
| border_terrier | 82.96% |
| australian_terrier | 77.78% |
| golden_retriever | 79.26% |
| old_english_sheepdog | 81.48% |
| samoyed | 94.07% |
| dingo | 77.78% |

### Most recent continuation

User asked to remove weight decay and continue with the previously discussed smaller learning rate. We explicitly interpreted that as0.00001 and communicated it before launch. The run finished.

Fresh AdamW with weight_decay0; lr0.00001 decaying over10 epochs; patience10; same augmentation, architecture, batch, split and seed. Starting public score80.9266%. Extra validation epochs:80.31,80.62,80.62,80.39,80.46,80.62,80.54,80.77,80.85,80.93%. Final train98.04%. No strict improvement, original state retained. This changes both regularization and continuation schedule, so it does not isolate the causal effect of removing weight decay.

## 6. Verification, archives and hashes

Completed selected checkpoints passed `verify_pipeline.py`: official labels, raw-pixel preprocessing parity, repeatability, single/multiple/empty batches, input validation, and fresh-process CPU loading. Split equality and recorded SHA-256 agreement were checked for recent experiments. Public-interface full validation reproduced saved scores. The continuation state tensors were compared with `torch.equal` against the initial checkpoint and matched exactly.

Best checkpoint SHA-256:

```text
c2edde1051dbe9e81d6cdde49610555c1527f0d2d724d1f898ae69e7f1265ce1
```

Current root and best-model exact source SHA-256:

```text
cf52f8e5f97a9f1d3ab20bf79fd02213f4811a21e4c51ef571020b41b6e86f2d
```

Continuation checkpoint SHA-256:

```text
84085af89188c7b9405ca55010ddddfb675ff4d2c2ed0e1c461827d4879146f7
```

The continuation file hash differs because saved metadata differs; state tensors are identical. Do not infer a new learned model from the file hash alone.

Preserved bundles:

- `saved_models/v3_wd001_8093/`: current80.93% best.
- `saved_models/v3_depth10_8000/`: earlier80.00% v3.
- `saved_models/v2_crop085_7892/`: earlier78.92% v2.
- `best_model/`: historical78.15% v2, intentionally not overwritten.

Ordinary run weights and data are ignored by Git. The three `saved_models` bundles above have explicit binary-weight exceptions and are actually tracked. Per-run exact source, metadata and recent verification records are tracked. `experiments/model_registry.json` records results and hashes. Keep trained weights, source and architecture metadata together.

Earlier authorized cleanup removed five disposable smoke directories after archiving their JSON records in `experiments/smoke_checks.json`. A redundant historical v2 continuation checkpoint was removed after tensor equality was established; its reference JSON points to original weights. Missing disposable artifacts are not evidence of training failure. Do not remove original data, coursework, full experiment records, or reference material.

## 7. Local Git history

Local repository on `master`, no configured Git upstream. User requested local version snapshots of different models.

| Commit | Work |
| --- | --- |
| `4c6fee8` | Preserve completed experiments through78.92% |
| `f86c1c5` | Prepare random erasing |
| `07ba934` | Record erasing78.22% |
| `3570a66` | Archive v2 best, including weights/source |
| `ec3d851` | Prepare v3 architecture |
| `4be6a58` | Complete v3 and archive80.00% |
| `7b10f66` | Prepare stronger weight decay |
| `eed77fa` | Complete/archive80.93% best |
| `837fa9a` | Record zero-decay continuation with no gain |

Tags include `v2-crop085-78.92`, `v2-erasing025-78.22`, `best-v2-7892-bundle`, `v3-prepared`, `v3-depth10-80.00`, and **`v3-wd001-80.93`**.

## 8. User preferences

The user prefers action over repeated confirmation, short clear progress updates, and explanations in plain language. For setup-only requests stop before training; for explicit launch requests launch after accelerator checks. Earlier compaction-related setup stops were superseded by later explicit training authorization. There is currently no authorization for an additional unrequested experiment.

### Update after the report: stronger weight decay (no gain)

| Run directory | Change | Best epoch | Best validation | Flip-averaged (measured separately) |
| --- | --- | ---: | ---: | ---: |
| `runs/v3_weight_decay_001` | wd 0.001 (current best) | 45 | 80.9266%, 1048/1295 | 81.8533%, 1060/1295 |
| `runs/v3_wd_01` | wd 0.01 | 39 | 79.4595%, 1029/1295 | 79.3822%, 1028/1295 |
| `runs/v3_wd_05` | wd 0.05 | 43 | 79.9228%, 1035/1295 | 81.0039%, 1049/1295 |

Both new runs: fresh 50 epochs, everything else identical, completed, public `predict` reproduced, splits identical. Weight decay 0.001 remains best. Flip averaging is not implemented in `predict`; the user will decide later.

## 9. Remaining work and possible directions

No new experiment is prepared or running beyond the completed ones above. Previously discussed ideas include testing an alternative optimizer such as SGD with momentum, repeating the strongest setup with a different seed, or other manually selected compliant changes. These are possibilities, not completed work or guaranteed improvements. Stronger augmentation by random erasing and simple continuation have already been tested and failed to help in these settings. Do not repeat them accidentally.

Reasonable next review: inspect class confusions, especially english_foxhound, and choose one deliberate experiment with a clear hypothesis. Retain supplied split for comparable results, or explicitly document any allowed resplit. Repeated tuning on one validation set and a single seed does not establish hidden-test accuracy. No hidden-test result or top-three placement is known.

Submission requirements:

- Deadline from `extra directions.txt`: **November8,2026,11:59pm**, Canvas; timezone not specified in local materials.
- Archive name `GroupX_Assignment.zip`, replacingX with actual group number; include code `.py`/`.ipynb` and trained `model.pth`/`.keras`.
- Contribution PDF must describe all members' responsibilities, specific contributions, percentages totaling100%, collaboration, and review/acknowledgment by all members. Obtain actual group names, IDs, percentages and acknowledgments; do not invent them.
- Need to confirm whether contribution PDF goes inside ZIP or separately on Canvas.
- Submission comment must list teammate names and IDs.
- One group member books evaluation for November9–22, PGH370, via instructor email/Teams. No slides/statistical analysis required. Do not send messages without explicit authorization.
- Before final packaging, stage exact chosen source and weights together, test fresh CPU loading/public inference from staging, and inspect archive contents. Do not submit automatically.

## 10. Useful commands and next-agent checks

```bash
# Local checks; do not train:
python3 verify_pipeline.py saved_models/v3_wd001_8093/model.pth
git status --short
git log -n 10 --oneline

```

Primary records: `AGENTS.md`, this report, per-run config/history/summary/verification files, per-experiment README, `experiments/model_registry.json`, and bundle manifests. Some older narrative setup documents retain historical phrases such as “prepared” or an older “best”; use the latest actual artifacts and this current state rather than treating those stale paragraphs as live process status.
