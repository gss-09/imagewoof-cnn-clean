# DogCNN-v3: prepared, not trained

Test a larger custom sequential CNN against the strongest measured DogCNN-v2 checkpoint: 78.9189% (1,022/1,295), epoch 47. This experiment changes both depth and width, so it measures their combined effect. A larger model may overfit; improvement is not assumed.

Five blocks use channels 32, 64, 128, 256, 256. Each block has two 3x3 Conv → BatchNorm → ReLU sequences and 2x2 max pooling. Pooling reduces 128x128 inputs to 64, 32, 16, 8, then 4 pixels per side. Finish with global average pooling, Flatten, Dropout 0.3, and Linear(256, 10). No pretrained weights, skip connections, or attention. Architecture ID: DogCNN-v3; 2,356,458 trainable parameters versus v2's 843,114.

Restore the winning augmentation: random resized crop scale 0.85–1.0, horizontal flip, color jitter 0.15; no random erasing. Keep official split (9,065/1,295), RGB128, normalization 0.5/0.5, seed42, AdamW lr0.001 weight decay0.0001, cosine50, label smoothing0.1, batch32, patience10, workers0, threads4. Initialize from scratch; do not use --init-checkpoint. V1/v2 checkpoints remain supported. CLI default remains v1; select v3 explicitly.

Prepared command (pending at the time):

```bash
python -u solution.py --architecture DogCNN-v3 --device cuda --output runs/depth_10conv --epochs 50 --batch-size 32 --lr 0.001 --weight-decay 0.0001 --patience 10 --seed 42 --workers 0 --threads 4
```

CPU inference checks are recorded in v3_checks.json. Temporary untrained weights were used for checkpoint round-trip and the isolated inference verifier; the trained best-v2 checkpoint also passed with the new code. No optimizer updates, smoke training, or full training were performed. accelerator memory, training speed, and learned accuracy remain unmeasured. Before launching, check a real augmented batch and forward/backward update as part of the authorized training setup, then save exact source in the new run directory.

Best measured weights and exact matching source are preserved in saved_models/v2_crop085_7892/, including a SHA-256 manifest. Local Git tag best-v2-7892-bundle includes that bundle's actual binary weights. Original run folders and the historical best_model/ bundle remain intact.
