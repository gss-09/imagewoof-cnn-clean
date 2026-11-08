# DogCNN-v3: stronger weight decay

Completed all 50 epochs. Best epoch 45: **80.9266% (1,048/1,295)**, +12 images / +0.9266 percentage points versus the previous 80.00% v3. Final epoch: training 98.18%, validation 80.54%. Reloaded public prediction reproduced the best score, CPU verifier passed, split matches the reference, and recorded source and checkpoint hashes match. Process exited. Exact artifacts saved locally in `runs/v3_weight_decay_001` and best bundle `saved_models/v3_wd001_8093`. Earlier bundles preserved.


Hypothesis: stronger weight decay may reduce overfitting. Previous v3 reached 97.77% final training accuracy, 79.38% final validation, and 80.00% best validation at epoch 45. Improvement is not guaranteed.

Change only AdamW weight decay from 0.0001 to 0.001. Keep ten-convolution DogCNN-v3, fresh initialization, seed42, supplied 9,065/1,295 split, crop0.85–1.0, flip, jitter0.15, no erasing, normalization0.5, batch32, lr0.001, cosine50, label smoothing0.1, patience10, workers0, threads4. CLI already supports this setting; no model or training-code change is needed. `planned_config.json` is a plan, not a measured run config.

Evaluation plan (completed): compare against **80.00% (1,036/1,295)**, confirm identical split, public prediction reproduction, per-class results, CPU pipeline verification and recorded hashes. Save exact source and checkpoint locally; update model registry, docs, and local Git. Preserve `saved_models/v3_depth10_8000/` even if this run is worse. Single-seed improvements are provisional. Do not launch SGD or another-seed experiments as part of this setup.
