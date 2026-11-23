# Imagewoof sequential CNN experiments

A custom sequential PyTorch CNN trained from scratch to classify ten Imagewoof breeds from 128 × 128 RGB images. The model uses only the supplied images and returns class IDs in the official order.

## Validation results

| DogCNN-v4 evaluation split | Top-1 accuracy | Correct / held out | Checkpoint |
| --- | ---: | ---: | --- |
| Original course validation split | **86.72%** | 1,123 / 1,295 | [Original-split best](saved_models/v4_sgd_wd001_8672/README.md) |
| 90/10 re-split | **85.91%** | 890 / 1,036 | [90/10 bundle](saved_models/v4_resplit9010_8591/README.md) |
| 95/5 re-split | **87.84%** | 455 / 518 | [95/5 bundle](saved_models/v4_resplit9505_8784/README.md) |
| 99/1 re-split | **89.52%** | 94 / 105 | [Checkpoint packaged for Group 20](saved_models/v4_resplit9901_8952/README.md) |

The original-split checkpoint scored 86.72% on CPU and 86.64% on accelerator. The re-splits pool the supplied training and validation data, then hold out different images. The 99/1 result comes from 105 images, so each prediction changes the score by about 0.95 percentage points. All four figures are validation results.

The earlier DogCNN-v3 checkpoint reached **80.93%** on the original validation split. Follow the [model development timeline](MODEL_TIMELINE.md) from the first commit to the final checkpoint. The [Group 20 package notes](submission/README.md) describe the final verification.

## Files and responsibilities

| File / folder | Purpose |
| --- | --- |
| `saved_models/` | Versioned checkpoint bundles with matching inference code and training records |
| `best_model/` | Historical v2 bundle that reached 78.15% |
| `experiments/PROCESS.md` | Saved training process, results, verification, and cleanup record |
| `solution.py` | Four CNN architectures, training CLI, checkpoint loading, and grader prediction interface |
| `verify_pipeline.py` | Dataset label checks and CPU inference checks, including a fresh isolated process |
| `train_mps.command` | Terminal launcher for the default v1 baseline |
| `experiments/depth_8conv.md` | Prepared v2 design, checks, launch command, and evaluation plan |
| `runs/baseline_full/` | Local baseline checkpoint, logs, and plots. Configuration and summary are tracked in Git |
| `requirements.txt` | Python dependencies |
| `AGENTS.md` | Instructions for agents working in this folder |

## Project graph

The project includes the `$graphify` skill. With the [Graphify CLI](https://github.com/Graphify-Labs/graphify) installed, refresh the local code index from the repository root with `graphify extract . --code-only`. The index includes the main training and inference code, the final recipe source, and the verification utilities.

## Model and preprocessing

In DogCNN-v1, three blocks each contain two `3×3 Conv → BatchNorm → ReLU` sequences followed by `2×2 MaxPool`. Channel counts are 32, 64, and 128. Global average pooling, dropout (0.3), and a ten-output linear layer complete the model: **288,746 trainable parameters**. Cross-entropy operates on logits; inference returns their argmax in the official class order.

**Channels** are feature maps: each convolution learns filters that produce separate maps of useful image patterns. The input has three channels (red, green, blue); a convolution with 32 output channels produces 32 learned feature maps. Their meaning is learned during training. Channel count describes the width of a layer; the number of convolutional layers describes depth.

| Architecture | Convolution layers | Channels per block | Spatial sizes after pooling | Parameters |
| --- | ---: | --- | --- | ---: |
| `DogCNN-v1` | 6 | 32, 64, 128 | 64×64, 32×32, 16×16 | 288,746 |
| `DogCNN-v2` | 8 | 32, 64, 128, 192 | 64×64, 32×32, 16×16, 8×8 | 843,114 |
| `DogCNN-v3` | 10 | 32, 64, 128, 256, 256 | 64×64, 32×32, 16×16, 8×8, 4×4 | 2,356,458 |
| `DogCNN-v4` | 10 | 64, 128, 256, 512, 512 | 64×64, 32×32, 16×16, 8×8, 4×4 | 9,413,066 |

Both convolutions in each block produce the listed number of channels. Adding the fourth block increases depth and capacity together; the completed run improved best validation accuracy from 70.50% to 78.15%.

Training augments images with random resized crops, horizontal flips, and mild color jitter. The final v4 recipe uses crops covering 35–100% of the source image, SGD with Nesterov momentum, a five-epoch warm-up followed by cosine decay, exponential moving average weights, and seed 42. Validation and inference use single-view RGB inputs normalized to `[-1, 1]`.

The training CLI defaults to the v1 baseline and the original course split. Use `--architecture DogCNN-v4` to select v4 and `--resplit` to evaluate a new holdout from the supplied data. Checkpoint loading selects the architecture recorded in the weights.

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

The baseline checkpoint is in the local run folder. See the [recorded summary](runs/baseline_full/summary.json) for per-breed accuracy.

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

## Inference and submission

`solution.py` exposes `load_model` and `predict`. The model loads on CPU in evaluation mode. `predict` accepts RGB `uint8` NumPy arrays of shape `(N, 128, 128, 3)` and returns `int64` class IDs of shape `(N,)`. It normalizes internally and processes up to 32 images per batch.

```python
from solution import load_model, predict

model = load_model("saved_models/v4_resplit9901_8952/model.pth")
predictions = predict(model, images)  # images is your RGB uint8 NumPy batch
```

The locally prepared Group 20 ZIP contains the selected checkpoint and its matching submission code. The package verification is recorded in the [submission notes](submission/README.md).

## Git history

Git tracks source code, experiment records, and the model weights explicitly included in versioned bundles under `saved_models/`. The datasets, submission ZIP, and ordinary run weights are stored locally. See the [model registry](experiments/model_registry.json) for checkpoint provenance.
