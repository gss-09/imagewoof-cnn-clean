# DogCNN-v3: completed, 80.00% validation

Test a larger custom sequential CNN against the strongest measured DogCNN-v2 checkpoint: 78.9189% (1,022/1,295), epoch 47. This experiment changes both depth and width, so it measures their combined effect. A larger model may overfit; improvement is not assumed.

Five blocks use channels 32, 64, 128, 256, 256. Each block has two 3x3 Conv → BatchNorm → ReLU sequences and 2x2 max pooling. Pooling reduces 128x128 inputs to 64, 32, 16, 8, then 4 pixels per side. Finish with global average pooling, Flatten, Dropout 0.3, and Linear(256, 10). No pretrained weights, skip connections, or attention. Architecture ID: DogCNN-v3; 2,356,458 trainable parameters versus v2's 843,114.

Restore the winning augmentation: random resized crop scale 0.85–1.0, horizontal flip, color jitter 0.15; no random erasing. Keep official split (9,065/1,295), RGB128, normalization 0.5/0.5, seed42, AdamW lr0.001 weight decay0.0001, cosine50, label smoothing0.1, batch32, patience10, workers0, threads4. Initialize from scratch; do not use --init-checkpoint. V1/v2 checkpoints remain supported. CLI default remains v1; select v3 explicitly.

Command used for the completed run (record only; do not relaunch into the existing output directory):

```bash
python -u solution.py --architecture DogCNN-v3 --device cuda --output runs/depth_10conv --epochs 50 --batch-size 32 --lr 0.001 --weight-decay 0.0001 --patience 10 --seed 42 --workers 0 --threads 4
```

Pre-launch CPU inference checks are recorded in v3_checks.json. Temporary untrained weights were used for checkpoint round-trip and the isolated inference verifier; the trained best-v2 checkpoint also passed with the new code. At preparation time no optimizer updates or training had been performed. The subsequent authorized CUDA batch check and full run are recorded below; exact source was saved in the run directory.

Best measured weights and exact matching source are preserved in saved_models/v2_crop085_7892/, including a SHA-256 manifest. Local Git tag best-v2-7892-bundle includes that bundle's actual binary weights. Original run folders and the historical best_model/ bundle remain intact.

## Completed result

Authorized run completed all 50 epochs. Best epoch 45: **80.00% (1,036/1,295)**; public prediction reproduced the score. Improvement over v2 gentler crop: 14 images / 1.0811 percentage points, one seed only. Final epoch: train 97.77%, validation 79.38%; the epoch-45 checkpoint is retained. This gap suggests overfitting; more epochs are not automatically useful. No training remains active. CPU pipeline verifier passed; checkpoint and source hashes verified. Exact source, config, history, summary, checkpoint, and verification are in `runs/depth_10conv/`. Best bundle is `saved_models/v3_depth10_8000/`, intentionally committed including weights.

Total epoch time: 17.79 minutes. CUDA disposable forward/backward setup check passed before launch; see `v3_training_setup.json`. Historical preparation statements above describe the pre-launch state.
