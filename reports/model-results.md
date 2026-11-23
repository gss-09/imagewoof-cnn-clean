# Imagewoof model results

Top-1 accuracy is measured on the validation split shown. The original course split held out 1,295 images. The later splits pooled the supplied training and validation images and re-split them by breed with seed 42.

| Checkpoint | Validation split | Correct | Top-1 | Selected epoch |
| --- | --- | ---: | ---: | ---: |
| DogCNN-v1 baseline | Original | 913 / 1,295 | 70.50% | 46 |
| DogCNN-v2 | Original | 1,012 / 1,295 | 78.15% | 45 |
| DogCNN-v2 with gentler crops | Original | 1,022 / 1,295 | 78.92% | 47 |
| DogCNN-v2 with random erasing | Original | 1,013 / 1,295 | 78.22% | 41 |
| DogCNN-v3 | Original | 1,036 / 1,295 | 80.00% | 45 |
| DogCNN-v3, AdamW weight decay 0.001 | Original | 1,048 / 1,295 | 80.93% | 45 |
| DogCNN-v3, AdamW weight decay 0.01 | Original | 1,029 / 1,295 | 79.46% | 39 |
| DogCNN-v3, AdamW weight decay 0.05 | Original | 1,035 / 1,295 | 79.92% | 43 |
| DogCNN-v4, 150 epochs with AdamW | Original | 1,081 / 1,295 | 83.47% | 83 |
| DogCNN-v4, SGD at 0.05 | Original | 1,115 / 1,295 | 86.10% | 89 |
| DogCNN-v4, SGD with weight decay 0.001 | Original | 1,123 / 1,295 | **86.72%** | 123 |
| DogCNN-v4, 70/30 | 70/30 re-split | 2,609 / 3,109 | 83.92% | 102 |
| DogCNN-v4, 90/10 | 90/10 re-split | 890 / 1,036 | 85.91% | 83 |
| DogCNN-v4, 95/5 | 95/5 re-split | 455 / 518 | 87.84% | 126 |
| DogCNN-v4, 95/5 with a 200-epoch schedule | 95/5 re-split | 453 / 518 | 87.45% | 162 |
| DogCNN-v4, 99/1 | 99/1 re-split | 94 / 105 | **89.52%** | 113 |

The 99/1 package used DogCNN-v4 with ten convolutions and 9,413,066 parameters. Training used batch size 32, SGD at 0.05 with Nesterov momentum 0.9 and weight decay 0.001, five warm-up epochs, and cosine decay over 150 epochs. Augmentation used random crops covering 35–100% of image area, horizontal flips, and color jitter. The checkpoint preserves the exponential moving average weights from epoch 113. Public inference reproduced 94 of 105 correct predictions. Each image changes that score by about 0.95 percentage points.

The all-data runs have no held-out score. See the [experiment timeline](../MODEL_TIMELINE.md) for the full run record and the [model registry](../experiments/model_registry.json) for checkpoint provenance.
