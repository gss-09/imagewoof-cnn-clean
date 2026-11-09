# Agent guidance for this folder

## Purpose and scope

Help complete COSC 4368 HW2: design, train, validate, and package a custom sequential CNN that classifies the ten Imagewoof dog breeds. Optimize for generalization and a reliable grading interface while following the course rules.

This file applies throughout this folder. Read the current user request before acting: documentation or review requests do not authorize starting a training run or completing the entire assignment.

## Read these sources first

1. `HW2-1.pdf`: assignment requirements, architecture constraints, grading interface, and submission format.
2. `extra directions.txt`: instructor clarifications and the additional contribution report requirement. Use these clarifications for the topics they address; they supplement the PDF.
3. Both `classes.txt` files: authoritative numeric label mapping.
4. Relevant lecture material when implementing or explaining the model:
   - `lectures/lecture4_MLP (1).pptx` (97 slides): activations, losses, backpropagation, regularization, initialization, optimizers, and batch normalization.
   - `lectures/lecture5_CNN.pptx` (72 slides): convolutions, pooling, spatial dimensions, small filters, and CNN design.
   - `lectures/lecture6_RNN.pptx` (41 slides): sequence models, RNNs, LSTMs, and GRUs; background rather than an architecture to use here.
   - `lectures/lecture7_transformer.pptx` (68 slides): attention and transformers; background rather than permission to use these methods.

Lectures discuss techniques such as residual connections, transfer learning, and architecture search that this assignment prohibits. Follow the assignment constraints when choosing an implementation.

## Folder inventory

The folder now contains course materials, supplied data, a working PyTorch implementation, a completed baseline, and a prepared second architecture. Recheck run artifacts before reporting execution status.

| Path | Contents / intended use |
| --- | --- |
| `HW2-1.pdf` | Five-page assignment specification |
| `extra directions.txt` | Tuning and split clarifications, deadline, live evaluation, contribution report |
| `lectures/` | Four reference decks listed above |
| `imagewoof_train/classes.txt` | Official class IDs |
| `imagewoof_train/train/<breed>/*.png` | 9,065 provided training images |
| `imagewoof_val/classes.txt` | Same official class IDs |
| `imagewoof_val/val/<breed>/*.png` | 1,295 provided validation images |
| `solution.py` | Self-contained training CLI and grader interface; selectable `DogCNN-v1` and `DogCNN-v2` |
| `verify_pipeline.py` | Dataset labels, preprocessing, checkpoint portability, and isolated CPU inference checks |
| `requirements.txt` | Python dependencies |
| `train_mps.command` | MPS Terminal launcher; currently selects default v1, not v2 |
| `README.md` | Usage, architecture explanation, results, and current status |
| `runs/baseline_full/` | Completed v1 checkpoint, logs, results report, plots, and reference comparison |
| `experiments/depth_8conv.md` | Prepared v2 experiment and launch/evaluation plan |

## Current experiment state and user directions

- **SUBMITTED (Nov 8, 2026, by the user): 90/10 re-split model `runs/r8_v4_resplit9010`, EMA best epoch 83, 85.91% (890/1,036)**, bundle `saved_models/v4_resplit9010_8591_SUBMITTED/` (sha 1a034107…, tag `submitted-v4-resplit9010-85.91`). User later clarified the TA did not require best-by-validation selection; in the PDF it is a Helpful tip. User now plans to resubmit the 99/1 model (saved_models/v4_resplit9901_8952/) if Canvas allows. Exclude outside test images and runs r13/r14 from model selection. Remaining: contribution PDF, Canvas comment with names/IDs, live-evaluation booking.
- Rounds 11–12 (Nov 8): 200 epochs on 95/5 gave 87.45% @162 (worse than 87.84% at 150; `runs/r11_v4_resplit9505_ep200`); 100/0 200 epochs `runs/r11_v4_trainval_ep200` unscored, not preferred over ep125. 99/1 re-split 150 epochs: **89.52% (94/105) @113**, very noisy, bundle `saved_models/v4_resplit9901_8952/` (tag `v4-resplit9901-89.52`). Submission choice still pending. No training active.
- Rounds 7–10 (Nov 8; `experiments/round7..round10/README.md`), all the 86.72% recipe with seeded stratified re-splits of pooled train+val via `--resplit` (source ad63459a…): 70/30 83.92% (2,609/3,109) @102; 90/10 85.91% (890/1,036) @83; 95/5 **87.84% (455/518) @126** (`saved_models/v4_resplit9505_8784/`, tag `v4-resplit9505-87.84`, `eval_split.py` reproduces it on CPU); 100/0 full 150 epochs `runs/r9_v4_trainval_ep150` (unscored; identical to r6 through epoch 125). New bundles: `saved_models/v4_trainval100_ep125/` (r6) and `v4_trainval100_ep150/` (r9), both unscored, CPU verifier passed. Top-2 diagnosis of 86.72% model: top-2 93.13%, beagle↔English foxhound 28 of 172 errors (`experiments/top2_diagnosis.*`); prior-based logit shift toward foxhound lowered overall accuracy. PDF check: "keep val close to train / save best by val" are Helpful tips; "predict reproduces your validation accuracy" is a Before-submitting checkbox. Recommended submission: 95/5 (every checkbox literal) or 100/0 ep125 (most data); user decision pending, plus group number, names/IDs, contribution PDF. Do not download or use original Imagewoof/ImageNet even for testing (contains hidden test set). No training active.
- Round 6 (Nov 8, `experiments/round6/README.md`): wd 0.002 gave 85.64% @113 (no gain). `runs/r6_v4_trainval_ep125` = 86.72% recipe trained on train+val via new `--train-on-val --stop-epoch 125` (unscored; CPU verifier passed; checkpoint sha 508cf0ac…). Best remains `saved_models/v4_sgd_wd001_8672/`. Pending user decision: submit the measured best or the train+val retrain. No training active.
- **Current best (November 8, 2026): `saved_models/v4_sgd_wd001_8672/`**, tag `v4-sgdwd001-86.72`. Run `runs/r5_v4_sgd_wd001`: same as the 86.10% recipe but SGD weight decay 0.001; EMA best epoch 123: 86.7181% (1,123/1,295) on CPU, 86.64% (1,122) accelerator public `predict`; CPU verifier passed; flip 86.49% (1,120, does not help). Round 5 (`experiments/round5/README.md`): 100-epoch schedule gave 84.63% (memorized sooner). No training active (round 5 finished Nov 8). Previous best: `saved_models/v4_sgd005_8610/`, tag `v4-sgd005-86.10`. Run `runs/r4_v4_sgd_lr005`: DogCNN-v4, 150 epochs, crop 0.35, SGD Nesterov momentum 0.9 lr 0.05 wd 0.0005, 5 warm-up epochs + cosine, EMA 0.999, best epoch 89: 86.1004% (1,115/1,295) via public `predict`; CPU verifier passed; flip averaging measured separately 86.4093% (1,119). Round 4 (`experiments/round4/README.md`): SGD lr 0.1 gave 85.64% (flip 85.71%). Switching AdamW to SGD gave +2.6 points. Previous best: `saved_models/v4_long150_8347/`, tag `v4-long150-83.47`. Run `runs/r2_v4_long150`: DogCNN-v4 (64/128/256/512/512, 9.4M params), 150 epochs, crop 0.35–1.0, AdamW lr 0.001 wd 0.001, EMA 0.999 weights saved, best epoch 83: 83.4749% (1,081/1,295) via public `predict`; CPU verifier passed. Flip averaging measured separately 84.4015% (1,093); `predict` still single view. Round 1 (`experiments/round1/README.md`): r1_v3_control 80.77% (noise check), r1_v4_wide 50 epochs 81.39%, r1_v3_long150 82.70% (flip 82.86%). Lesson: longer schedule + stronger crops gave the big gain; width added ~+0.8. Round 3 (`experiments/round3/README.md`, Nov 8): r3_v4_aug_strong (crop 0.25 + rotation ±10°) 82.63%, flip 83.47%; r3_v4_seed43 (same recipe, seed 43) 82.93%, flip 83.32%. No new best; seed noise ~0.5 point; stronger aug hurt. Pending user decisions: flip averaging in `predict`, train+val retrain, Mixup/CutMix compliance. No training active (round 4 finished Nov 8).

- Stronger weight-decay runs completed November 7, 2026: DogCNN-v3, fresh 50 epochs, all settings as the 80.93% run except AdamW weight decay. wd 0.01 (`runs/v3_wd_01`): best epoch 39, 79.4595% (1,029/1,295). wd 0.05 (`runs/v3_wd_05`): best epoch 43, 79.9228% (1,035/1,295). Both reproduced through public `predict`; splits identical; recorded hashes match; weights ignored by Git. No improvement: wd 0.001 remains best of 0.0001/0.001/0.01/0.05; keep `saved_models/v3_wd001_8093/`. No training active. Horizontal-flip averaging measured separately (scratch script, not implemented in `predict`): 80.93% bundle 81.8533% (1,060), wd 0.05 81.0039%, wd 0.01 79.3822%. User is deciding later whether to add flip averaging; do not change `predict` without being asked. See `experiments/v3_wd_01/README.md`, `experiments/v3_wd_05/README.md`.

- Zero-weight-decay continuation completed 10 extra epochs, output `runs/v3_no_decay_continue`, log `setup/v3_no_decay_continue.log`, former Python PID2803045 exited. Fresh AdamW lr0.00001 cosine10, wd0 from 80.9266% checkpoint. No gain: extra epoch10 matched80.9266%; selected best remains initial epoch0 with exactly identical state tensors to original epoch45. Public prediction and CPU verifier passed; same split and recorded hashes confirmed. Keep best bundle `saved_models/v3_wd001_8093/`. No training active. See `experiments/v3_no_decay_continue/README.md`.

- Weight-decay experiment completed all 50 epochs: `runs/v3_weight_decay_001`, DogCNN-v3 AdamW weight decay 0.001, all other settings unchanged from previous v3. Best epoch 45: 80.9266% (1,048/1,295), +12 images over previous 80.00%. Public prediction reproduced score, CPU checks passed, identical split and recorded hashes confirmed. New best bundle `saved_models/v3_wd001_8093/`, intentionally including binary weights in Git, tag `v3-wd001-80.93`. No training active; previous Python PID 2800850 exited. Preserve earlier bundles. User asked about extra epochs: final validation 80.54% after peak at45, final train98.18%; no extension launched. See `experiments/v3_weight_decay_001/README.md`.

- Authorized DogCNN-v3 training completed all 50 epochs. Best epoch 45: 80.00% (1,036/1,295), +14 images / +1.0811 percentage points versus v2 gentler crop. Reloaded public prediction reproduced the score; CPU pipeline verifier passed; checkpoint and source hashes verified. No training remains active. Run: `runs/depth_10conv`, exact source saved there. Current best bundle: `saved_models/v3_depth10_8000/`, including binary weights intentionally tracked in Git, tag `v3-depth10-80.00`. Ten convolutions, channels 32/64/128/256/256, five pools, no erasing; all other planned settings retained. Preserve earlier v2 and historical bundles. See `experiments/depth_10conv.md`.

- Random erasing completed all 50 epochs, output `runs/random_erasing_025`, log `setup/random_erasing_025.log`. Best epoch 41: 78.2239% (1,013/1,295), public prediction reproduced; CPU verifier passed and local download hashes match. It underperformed gentler crop by 9 images. No training is active. Settings: p=0.25, scale=(0.02, 0.12), ratio=(0.5, 2.0), value=0 after normalization, training only. See `experiments/random_erasing.md`. Preserve the completed CUDA 78.9189% checkpoint. User requested local Git snapshots for model progress: commit code and small metadata per experiment, keep ordinary run weights/data ignored (explicit best bundle exception above), record checkpoint hashes in `experiments/model_registry.json`, and save exact run source before editing. Never imply Git contains ignored weights.

- The fresh v2 run used a minimum crop area of 0.85, output `runs/gentler_crop_085`, log `setup/gentler_crop_085.log`. This run completed all 50 epochs: best epoch 47, 78.9189% (1,022/1,295), reproduced through public prediction. Checkpoint and exact training source are copied locally to `runs/gentler_crop_085/`. The subsequent random-erasing run also completed (78.22%); no training is active. See `experiments/gentler_crop.md`. Preserve `best_model/` as the historical 78.15% bundle with its original code and hashes.

- The user requested saving the best model and process and removing unnecessary files. `best_model/` holds exact copies of the selected v2 weights, matching code, dependencies, config/summary, and a SHA-256 manifest. See `experiments/PROCESS.md` for the consolidated workflow. All five disposable smoke directories were removed after archiving their complete JSON records in `experiments/smoke_checks.json`. The continuation checkpoint was tensor-identical to the original v2 checkpoint and was removed as redundant; use `runs/depth_8conv/model.pth` or `best_model/model.pth`, with provenance in `runs/depth_8conv_continue/checkpoint_reference.json`. Full run logs, both original trained checkpoints, source data, course materials, and reference files remain. Do not infer missing smoke directories or continuation weights indicate failed training. Keep the saved model bundle synchronized deliberately if code changes; its manifest covers the saved training artifacts.

- The full v1 baseline completed 50 epochs on MPS with seed 42. Best validation accuracy was **70.50% (913/1,295), epoch 46**, reproduced through `predict`. Its checkpoint is `runs/baseline_full/model.pth`; preserve it.
- v1 has six convolutions with block channels 32, 64, 128 and 288,746 parameters. v2 extends our baseline with a fourth two-convolution block at 192 channels, totaling eight convolutions and 843,114 parameters. Both remain sequential and use max pooling after each pair.
- v2 passed CPU forward, checkpoint round-trip, and isolated inference checks with temporary untrained weights. The trained baseline also passed the verifier after the architecture change. The subsequent two-epoch v2 MPS smoke run passed, including gradient updates and reloading (`runs/smoke_depth_8conv`); its 10% smoke accuracy is not a full-run result.
- The authorized full v2 run completed all 50 epochs on MPS on November 7, 2026 in `runs/depth_8conv`: best **78.15% (1,012/1,295), epoch 45**, reproduced through public `predict`; CPU portability verifier passed. Epoch time totaled 34.77 minutes. Split matches the baseline exactly; improvement is 7.64 percentage points. Preserve both checkpoints. The authorized MPS continuation in `runs/depth_8conv_continue` completed: fresh AdamW at 0.0001, cosine over at most 20 extra epochs, patience seven, same split and augmentation. It stopped after seven epochs (5.02 minutes) without improvement; best extra epoch was 77.22%. The saved weights exactly match the starting epoch-45 checkpoint, retaining 78.15% at continuation epoch 0, reproduced through full public prediction. No training process remains active. See `experiments/continuation.md`.
- Keep the implementation independent of third-party assignment solutions.
- The supplied train/validation split and inference preprocessing remain unchanged. `predict` uses a single view; horizontal-flip averaging was measured separately and is not implemented in our prediction path.
- MPS works on this Mac outside the restricted agent sandbox. A false availability result inside the sandbox does not establish a broken PyTorch installation. Use the approved execution environment or normal Terminal for accelerator work; request escalation when needed rather than reinstalling dependencies or silently using CPU.

All 10,360 local PNG files were checked with Pillow's image verification during the initial review: each is RGB, 128 × 128, with no verification errors. `.DS_Store` files are macOS metadata; ignore them in loaders and submission archives.

The full course dataset has 12,951 images. The remaining 2,591 images are the instructor's hidden test set and are not part of this workspace.

## Course constraints

- Build a custom deep **sequential CNN**, initialized and trained from scratch.
- Use only the permitted standard layers: Conv2D, pooling (including global average pooling), BatchNorm, Dropout, Flatten, Dense, and activation functions.
- Do not use pretrained models or weights, transformers, attention, residual/skip connections, dense blocks, or other architectures outside the sequential requirement.
- Use PyTorch or TensorFlow/Keras. The specification calls for a accelerator framework; inference must also work on CPU. Check actual hardware and dependencies before choosing how to train.
- Train at 128 × 128 RGB. Do not assume higher-resolution test inputs.
- Use only the provided training and validation images. Do not download original Imagewoof, ImageNet, Imagenette, or other external training images; these can overlap the hidden test set.
- Never seek or use the hidden test set for development or model selection.
- Manual hyperparameter tuning is allowed, including learning rate, optimizer, batch size, layers, and augmentation. Do not use hyperparameter optimization or architecture search tools/libraries such as Optuna, Ray Tune, Keras Tuner, or DARTS.
- Reasonable augmentation is allowed: crops resized to 128 × 128, flips, rotation, color jitter, and random erasing, for example.
- Combining and re-splitting the supplied train/validation data, including cross-validation, is allowed. Record any new split and keep evaluation examples separate from training during model selection.
- Use fixed random seeds and report the final model's seed. Document any remaining nondeterminism.

## Class labels and data loading

The grader requires this exact output order:

| ID | Folder / class | Training images | Validation images |
| --- | --- | ---: | ---: |
| 0 | `shih_tzu` | 945 | 135 |
| 1 | `rhodesian_ridgeback` | 945 | 135 |
| 2 | `beagle` | 945 | 135 |
| 3 | `english_foxhound` | 562 | 80 |
| 4 | `border_terrier` | 944 | 135 |
| 5 | `australian_terrier` | 944 | 135 |
| 6 | `golden_retriever` | 945 | 135 |
| 7 | `old_english_sheepdog` | 945 | 135 |
| 8 | `samoyed` | 945 | 135 |
| 9 | `dingo` | 945 | 135 |

- Read the mapping from `classes.txt`; confirm the two copies agree.
- **Do not trust alphabetical folder labels.** PyTorch `ImageFolder` and similar loaders commonly assign alphabetical IDs. Explicitly map their targets to the official IDs, or consistently convert internal output IDs in `predict`. Verify this mapping end to end.
- Point loaders at `imagewoof_train/train` and `imagewoof_val/val`, not their parent directories containing `classes.txt`.
- Preserve the supplied images, labels, lecture decks, and assignment documents. Store transformed data and generated artifacts separately.
- English foxhound has substantially fewer examples. Review per-class accuracy as well as overall top-1 accuracy.
- Apply random augmentation to training examples only. Keep validation and ordinary inference preprocessing deterministic and consistent.
- If computing normalization statistics, derive them from the training partition and save the values used for inference.

## Implementation and experiment workflow

When the user requests implementation or training:

1. Inspect existing work and the available Python environment and accelerator. Preserve user changes. Record dependencies and runnable commands; avoid machine-specific absolute paths.
2. Start with a small compliant baseline. The assignment suggests 4–6 convolutional layers, pairs of 3 × 3 convolutions before pooling, BatchNorm, Dropout, and global average pooling. These are starting suggestions, not mandatory architecture settings.
3. Keep model construction, training, checkpoint loading, and prediction understandable and consistent. A self-contained `solution.py` is a useful submission convention; an `.ipynb` is also allowed by the specification.
4. Guard training and command-line execution with `if __name__ == "__main__":` in Python modules. Importing the submitted module must not train, download data, or require the local training dataset.
5. Check one data batch, label mapping, tensor shapes, loss, and a forward/backward update before a full run. Discuss substantial runtime or unavailable compute when it affects the requested work.
6. Keep a record of manually chosen experiments: seed, split, architecture, augmentation, normalization, optimizer, schedule, batch size, epochs, validation accuracy, and checkpoint location. Do not replace manual choices with an automated search framework.
7. Save the best checkpoint by validation accuracy. Keep architecture and preprocessing metadata sufficient to reproduce it; avoid overwriting the only good checkpoint during an experiment.
8. Separate smoke runs from completed training. Report measured results and actual execution status; never invent accuracy, training completion, or test results.

Use `runs/` for experiment logs/checkpoints and `submission/` for final staging. Keep caches and intermediate artifacts out of the final archive. Keep dependency and usage documentation current. A quick syntax check is `python3 -m py_compile solution.py verify_pipeline.py`; check the actual trained baseline interface with `python3 verify_pipeline.py runs/baseline_full/model.pth`. Neither command trains a model.

## Required inference interface

The test harness imports the code and calls only these functions. For a PyTorch submission, preserve the following signatures and behavior:

```python
def load_model(path="model.pth"):
    """Build the trained architecture, load weights, and return it in eval mode."""

def predict(model, images):
    """Accept RGB uint8 NumPy data shaped (N, 128, 128, 3).
    Return an integer NumPy array shaped (N,), with official class IDs 0–9.
    """
```

- Perform all inference normalization inside `predict`; the harness supplies raw pixels in the range 0–255.
- Use the exact architecture corresponding to the checkpoint. Load weights portably with appropriate device mapping; do not require CUDA to load a accelerator-trained checkpoint.
- Disable training behavior and gradients for inference. In PyTorch, use evaluation mode and `inference_mode` or `no_grad`.
- Convert NHWC input to NCHW for PyTorch. Return CPU NumPy integer IDs, not tensors, logits, probabilities, or class names.
- Preserve input order and use bounded internal batches so the whole test set need not fit in accelerator memory.
- Keep inference independent of local dataset paths, external downloads, and unsubmitted helper modules.
- Keras is allowed with `model.keras`; retain the same callable contract and raw-pixel input/output behavior with the matching loading default.

## Validation before handoff

- Import the submission module in a fresh process and confirm that no training begins.
- Reload the actual final checkpoint on CPU, and on accelerator when available; explicitly state when a device was unavailable for checking.
- Pass real validation images as `(N, 128, 128, 3)` RGB `uint8` arrays through the public `predict` function.
- Check output shape, integer dtype, ID range, and class order for a single image and multiple images. Handle an empty batch sensibly if supported.
- Confirm public-interface accuracy reproduces the recorded validation accuracy for the same checkpoint, split, and preprocessing.
- Verify that evaluation is deterministic and that changing inference batch size does not materially change predictions.
- Confirm the submitted architecture, weights, preprocessing, and dependencies agree. Inspect archive contents and test loading from the staged submission without access to development-only helpers.
- Run relevant checks after code changes; do not claim a full training run or device test based on a smoke test.

## Submission and live evaluation

- Deadline stated in the supplied materials: **Sunday, November 8, 2026, at 11:59 pm**, as shown on Canvas. No timezone is specified locally; do not invent one. The PDF says submitted files are final and cannot be changed afterward.
- One submission per group, using the same group as the Project.
- Required archive: `GroupX_Assignment.zip`, replacing `X` with the actual group number. Include the code file (`.py` or `.ipynb`) and trained `model.pth` or `model.keras`.
- The instructor additionally requires a brief **individual contribution report in PDF format covering all members**: responsibilities, specific contributions, estimated percentages totaling 100%, and how the group collaborated. All members must review and acknowledge it.
- The clarifications do not specify whether the contribution PDF belongs inside the archive or is uploaded separately. Prepare it as a separate deliverable and verify the submission placement in Canvas or with the user when packaging. Do not fabricate names, IDs, percentages, contributions, or acknowledgments.
- After submission, a comment must list every teammate's name and ID. Obtain actual values from the group.
- One group member books the in-person evaluation for November 9–22 by email or Teams. Location: PGH 370. Help prepare scheduling text if requested; do not contact the instructor without an explicit request to send it.
- No slides or statistical analysis are required for the live evaluation. Prepare the group to explain the architecture, augmentation, training choices, seed, and how to run the submitted model.
- Prepare and verify local deliverables within the user's requested scope. Upload or submit only when the user requests that action.
