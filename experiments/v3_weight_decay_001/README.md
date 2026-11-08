# DogCNN-v3: stronger weight decay, prepared only

User requested setup and an explicit stop before training. Do not launch until newly authorized. No optimizer updates or smoke training were performed for this experiment.

Hypothesis: stronger weight decay may reduce overfitting. Previous v3 reached 97.77% final training accuracy, 79.38% final validation, and 80.00% best validation at epoch 45. Improvement is not guaranteed.

Change only AdamW weight decay from 0.0001 to 0.001. Keep ten-convolution DogCNN-v3, fresh initialization, seed42, supplied 9,065/1,295 split, crop0.85–1.0, flip, jitter0.15, no erasing, normalization0.5, batch32, lr0.001, cosine50, label smoothing0.1, patience10, workers0, threads4. CLI already supports this setting; no model or training-code change is needed. `planned_config.json` is a plan, not a measured run config.

After completion: compare against **80.00% (1,036/1,295)**, confirm identical split, public prediction reproduction, per-class results, CPU pipeline verification and recorded hashes. Save exact source and checkpoint locally; update model registry, docs, and local Git. Preserve `saved_models/v3_depth10_8000/` even if this run is worse. Single-seed improvements are provisional. Do not launch SGD or another-seed experiments as part of this setup.
