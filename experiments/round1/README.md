# Round 1: speed, run-to-run noise, longer training with stronger crops, wider network

Prepared November 7, 2026. Not launched until `experiments/round1/launch.sh` is run.

Reference: `runs/v3_weight_decay_001`, DogCNN-v3, 80.93% (1,048/1,295), raw weights, single view.

## Observations motivating this round

- Training accuracy reaches ~98% while validation plateaus at ~80% and validation loss stops falling after ~epoch 40: the network memorizes the weakly augmented training set (crops cover 85–100% of the image).
- Last-10-epoch validation std is 0.3–0.7 percentage points within a run; run-to-run noise is likely about ±1 point. Recent single-change differences (wd 0.0001 / 0.001 / 0.01 / 0.05 gave 80.00 / 80.93 / 79.46 / 79.92%) are within that band, so small tweaks cannot be told apart from luck.
- Single-process image loading (`--workers 0`) limited training throughput.
- Largest error: beagle ↔ English foxhound, 30 of 247 validation errors.

## Code changes (all opt-in; defaults reproduce earlier runs bit for bit, checked by a CPU smoke run)

- `--crop-min`: RandomResizedCrop lower area bound (default 0.85).
- `--architecture DogCNN-v4`: same 10-convolution sequential layout as v3, every block twice as wide (64/128/256/512/512, 9,413,066 parameters).
- `--ema-decay`: exponential moving average of weights and BatchNorm buffers. The averaged copy is the same sequential CNN; when enabled it is evaluated and saved, and the raw-weight validation curve is still logged (`validation_raw`).
- `--log-flip`: logs flip-averaged validation accuracy each epoch; does not affect training, selection, or `predict`.
- `--optimizer sgd`, `--momentum`, `--warmup-epochs`: available for a later round.
- `--workers N` already existed; DataLoader workers are seeded from the fixed generator.

## Runs (all seed 42, supplied split, AdamW lr 0.001, wd 0.001, batch 32, label smoothing 0.1, dropout 0.3, EMA 0.999 logged, workers 6)

| Run | accelerator | Change vs reference | Question |
| --- | --- | --- | --- |
| `r1_v3_control` | 0 | none except workers 6 (new random draw) and EMA | Speedup; how big is run-to-run noise; does EMA help? |
| `r1_v3_long150` | 0 | 150 epochs, crop 0.35–1.0 | Does stronger cropping with a longer schedule reduce memorization? |
| `r1_v4_wide` | 1 | 2× wider network | Is the network capacity-limited? |

Compare raw-weight best (from `validation_raw`) against 80.93% for single-change effects; EMA and flip numbers are extra information from the same runs.
