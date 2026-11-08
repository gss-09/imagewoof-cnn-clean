# Best measured model: DogCNN-v3, 80.93%

Exact epoch-45 checkpoint and training source from `runs/v3_weight_decay_001`, trained from scratch. AdamW weight decay 0.001; other settings match the earlier v3. Validation: 1,048/1,295, 12 more correct than the previous 80.00% model. Public prediction reproduced the score; CPU pipeline checks passed; recorded hashes match.

From this directory, import `load_model` and `predict` from `solution.py`, then call `load_model("model.pth")`. Inputs are raw RGB uint8 NumPy arrays shaped (N,128,128,3). Manifest records artifact hashes. Weights intentionally tracked in Git; earlier bundles preserved. This is validation accuracy.
