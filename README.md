# Imagewoof sequential CNN experiments

This baseline implements the six-convolution model discussed in the course workflow. It uses only the supplied images, starts from random weights, and assigns labels in `classes.txt` order. Read `AGENTS.md` and the assignment materials for the course constraints.

## Current status

- **DogCNN-v3 prepared, not trained:** ten convolutions; channels 32, 64, 128, 256, 256; successful gentler-crop augmentation, no random erasing. See [plan and checks](experiments/depth_10conv.md).

- **Random erasing completed:** best **78.22% (1,013/1,295), epoch 41**; public prediction and CPU portability checks passed. Lower than 78.92%, so retain the gentler-crop checkpoint. No training is active. See [results](experiments/random_erasing.md).

- **Gentler crop completed:** fresh CUDA v2 training reached **78.92% (1,022/1,295), epoch 47**, verified through public prediction. The checkpoint and exact source are saved locally in `runs/gentler_crop_085/`. See [experiment report](experiments/gentler_crop.md).

- **Saved best model:** [78.92% bundle](saved_models/v2_crop085_7892/README.md) includes exact source and weights, committed in Git under tag `best-v2-7892-bundle`. [best_model/](best_model/README.md) remains the historical 78.15% bundle. [Process report](experiments/PROCESS.md) records the full workflow and results. Disposable smoke runs were removed after archiving their JSON records in `experiments/smoke_checks.json`; full experiment records and both original trained checkpoints remain.

- **Continuation completed:** `runs/depth_8conv_continue` stopped after seven extra MPS epochs (5.02 minutes) without beating 78.15%. Best extra-epoch validation was 77.22%; the run retained the original best weights. It used fresh AdamW at 0.0001, cosine decay, and patience seven. See [continuation settings](experiments/continuation.md). `--init-checkpoint` loads weights with a fresh optimizer and schedule; it does not restore optimizer state.

- **Completed:** `DogCNN-v1` reached **70.50% validation accuracy** (913/1,295) at epoch 46, after a 50-epoch MPS run taking about 35 minutes of epoch time. Its saved checkpoint passed CPU portability and inference checks.
- **Completed November 7, 2026:** `DogCNN-v2` reached **78.15% validation accuracy** (1,012/1,295) at epoch 45, after 50 MPS epochs taking 34.77 minutes of epoch time. This improves on v1 by **7.64 percentage points** on the identical split. Reloaded public prediction reproduced the score and CPU portability checks passed. Checkpoint: `runs/depth_8conv/model.pth`; see [the experiment report](experiments/depth_8conv.md).

## Files and responsibilities

| File / folder | Purpose |
| --- | --- |
| `best_model/` | Selected weights, matching inference code, dependencies, and provenance |
| `experiments/PROCESS.md` | Saved training process, results, verification, and cleanup record |
| `solution.py` | Three CNN architectures, training CLI, checkpoint loading, and grader prediction interface |
| `verify_pipeline.py` | Dataset label checks and CPU inference checks, including a fresh isolated process |
| `train_mps.command` | Terminal launcher for the default **v1** architecture; does not select v2 |
| `experiments/depth_8conv.md` | Prepared v2 design, checks, launch command, and evaluation plan |
| `runs/baseline_full/` | Trained v1 checkpoint, configuration, history, results, and plots |
| `requirements.txt` | Python dependencies |
| `AGENTS.md` | Instructions for agents working in this folder |

## Model and preprocessing

Three blocks each contain two `3×3 Conv → BatchNorm → ReLU` sequences followed by `2×2 MaxPool`. Channel counts are 32, 64, and 128. Global average pooling, dropout (0.3), and a ten-output linear layer complete the model: **288,746 trainable parameters**. Cross-entropy operates on logits; inference returns their argmax in the official class order.

**Channels** are feature maps: each convolution learns filters that produce separate maps of useful image patterns. The input has three channels (red, green, blue); a convolution with 32 output channels produces 32 learned feature maps. Their meaning is learned during training. Channel count describes the width of a layer; the number of convolutional layers describes depth.

| Architecture | Convolution layers | Channels per block | Spatial sizes after pooling | Parameters |
| --- | ---: | --- | --- | ---: |
| `DogCNN-v1` | 6 | 32, 64, 128 | 64×64, 32×32, 16×16 | 288,746 |
| `DogCNN-v2` | 8 | 32, 64, 128, 192 | 64×64, 32×32, 16×16, 8×8 | 843,114 |

Both convolutions in each block produce the listed number of channels. Adding the fourth block increases depth and capacity together; the completed run improved best validation accuracy from 70.50% to 78.15%.

Images remain RGB at 128 × 128. The current training code uses random resized crops (scale 0.85–1.0; earlier MPS runs used 0.7–1.0), horizontal flips, mild color jitter, with no random erasing (the erasing experiment did not improve validation). Validation uses no augmentation. Both paths scale pixels to `[0, 1]` and normalize with mean and standard deviation `(0.5, 0.5, 0.5)`, yielding `[-1, 1]`. These are fixed constants, not statistics from external data.

The provided split stays intact. The default seed is 42. Defaults are AdamW, learning rate 0.001, weight decay 0.0001, label smoothing 0.1, a cosine learning-rate schedule, at most 50 epochs, and early stopping after ten epochs without improved validation accuracy. These are manually chosen baseline settings, not tuned results.

The default architecture remains `DogCNN-v1`, matching the completed baseline. `--architecture DogCNN-v2` selects our prepared eight-convolution extension: an additional pair of 3×3 convolutions with 192 channels followed by max pooling, for 843,114 parameters. Checkpoint loading automatically selects the saved architecture, so existing baseline weights remain supported. See [the experiment plan](experiments/depth_8conv.md) for the fixed settings, completed checks, and launch command. The full v2 run completed with 78.15% best validation accuracy.

## Environment

Use Python 3.11 and install the dependencies if needed:

```bash
python3 -m pip install -r requirements.txt
```

The initial execution environment already had these dependencies. The restricted shell could not access MPS, but an approved check outside the sandbox confirmed that MPS works on this M1 Pro. A two-epoch MPS smoke run completed successfully, including checkpoint reloading and public-interface accuracy verification. Fixed seeds are recorded; unsupported deterministic operations emit warnings, and identical results across different hardware are not guaranteed.

## Run a short pipeline check

From this folder:

```bash
python3 solution.py --smoke --device cpu --output runs/smoke_baseline
python3 verify_pipeline.py runs/smoke_baseline/model.pth
```

The smoke run uses 16 training and eight validation images per breed (160 and 80 total), for at most two epochs. It verifies gradient updates, validation, best-checkpoint saving, reloading, and accuracy through the public inference function. **Its weights and accuracy are only pipeline checks, not a trained assignment submission.**

Every run requires a new output directory to protect previous checkpoints. For another check, choose a name such as `runs/smoke_baseline_2`.

Initial CPU check: both epochs completed (about 6.5 seconds per epoch), with best validation accuracy 11.25% on the 80-image smoke partition. Reloading through `predict` reproduced 11.25%. All `verify_pipeline.py` checks passed, including labels for all 10,360 images and fresh-process CPU inference.

## Completed full baseline

`runs/baseline_full` completed all 50 epochs on MPS with seed 42, using all 9,065 training images and the supplied 1,295-image validation split. The best checkpoint reached **70.50% validation accuracy at epoch 46**. Reloading it and evaluating through the public `predict` interface reproduced 70.50% on the complete validation set. The final checkpoint also passed `verify_pipeline.py`, including isolated fresh-process CPU loading, class labels, preprocessing, and batch behavior.

The checkpoint is `runs/baseline_full/model.pth`. See [the results report](runs/baseline_full/RESULTS.md) for per-breed accuracy, major confusions, timings, and interpretation, and [the training curves](runs/baseline_full/training_curves.png) for the learning history. English foxhound was the weakest breed (32.50%); samoyed was strongest (87.41%). Hidden-test performance is unknown. The v2 run completed; a separate continuation completed without improvement.

## Train on the complete provided split

```bash
python3 solution.py --architecture DogCNN-v1 --output runs/baseline_repeat --epochs 50 --batch-size 32
```

This command starts another v1 run. The completed baseline already occupies `runs/baseline_full`; every new run needs an unused output directory. The completed v2 run used the command in [the experiment plan](experiments/depth_8conv.md).

Device selection automatically prefers CUDA, then MPS. Full training stops if neither accelerator is accessible; CPU training requires explicit `--device cpu`. Smoke runs can use CPU automatically. You can select `--device cuda` or `--device mps` to require a specific accelerator. An actual accelerator tensor operation is checked before training. Use `--data-root /path/to/data` on another machine, pointing at the directory containing both supplied dataset folders.

### Run on the Mac accelerator from Terminal

Double-click `train_mps.command` in Finder, or run it from a normal macOS Terminal:

```bash
./train_mps.command
```

It uses the Python installation tested here, requires MPS, checks an actual accelerator operation, runs up to 50 epochs with early stopping, prevents idle sleep, and then verifies CPU checkpoint portability. Each launch creates a timestamped run directory. It fails instead of falling back to hours of CPU training. Press Control-C in Terminal to stop; the best checkpoint saved so far remains available.

accelerator access diagnosis: Apple's `MTLCreateSystemDefaultDevice` returned no device inside the restricted agent shell. After the user changed the approval settings, an approved outside-sandbox tensor check returned `MPS available: True` and the actual model trained on MPS. No PyTorch reinstall was needed. The `runs/smoke_mps` run used 160 training and 80 validation images: epoch one took 6.2 seconds including initial accelerator work, and epoch two took 0.7 seconds. Its best accuracy was 10%, reproduced after checkpoint reloading; these are smoke results, not full training results. Agent-run MPS commands require approval to execute outside the sandbox.

Each run records:

- `config.json`: seed, settings, software version, device, architecture, and preprocessing.
- `split.json`: exact image paths used for each partition.
- `history.json`: epoch losses, accuracies, confusion matrices, and timings.
- `model.pth`: best checkpoint by validation accuracy, including class mapping and metadata.
- `summary.json`: best epoch, overall and per-class validation accuracy, and public-interface accuracy after reloading.

The script checks public-interface accuracy against the best checkpoint on the entire validation partition used in that run. Run `verify_pipeline.py` on the full checkpoint as well. It checks all dataset labels, preprocessing parity, repeatability, batch behavior, malformed inputs, and inference in a fresh process with only the code and weights available.

## Grader interface and eventual submission

`solution.py` is self-contained for inference. Importing it does not train or load the dataset. `load_model(path="model.pth")` restores the checkpoint on CPU in evaluation mode. `predict(model, images)` accepts RGB `uint8` NumPy arrays of shape `(N, 128, 128, 3)` and returns NumPy `int64` IDs shaped `(N,)`. It normalizes internally and processes at most 32 images per internal batch. To use an accelerator, move the loaded model to that device before calling `predict`.

After actual training and validation, stage `solution.py` with the chosen run's `model.pth` for `GroupX_Assignment.zip`. Do not substitute smoke weights. The contribution PDF, group number, names, IDs, and member acknowledgments still require the group's real information. No submission archive is created by these commands.

## Local version history

Git stores code and small experiment records; datasets and ordinary run weights stay outside Git. The explicitly saved best bundle in `saved_models/v2_crop085_7892/` includes its binary weights in Git. Completed checkpoints remain in their local `runs/` folders and are identified by SHA-256 in `experiments/model_registry.json`. Git backs up the named best bundle; other run weights remain outside Git.

`git log --oneline` shows progress. Tag `v2-crop085-78.92` preserves the completed gentler-crop code before random erasing. Use `git show v2-crop085-78.92:solution.py` to inspect it without replacing current files. The first commit also preserves earlier run metadata and the original 78.15% bundle source; earlier commits were not reconstructed.

DogCNN-v3 adds a fifth block and widens block four: channels `32, 64, 128, 256, 256`, ten convolutions and five pooling stages. Select `--architecture DogCNN-v3`; default remains v1. Current code restores augmentation from the winning gentler-crop run. V3 has no trained accuracy yet.
