# Best measured model: DogCNN-v3, 80.00%

Exact epoch-45 checkpoint and training Python source from `runs/depth_10conv`, trained from scratch. Validation: 1,036/1,295; public prediction reproduced 80.00%; CPU portability checks passed. No training remains active. This is validation accuracy, not a hidden-test measurement.

From this directory, import `load_model` and `predict` from `solution.py`, then call `load_model("model.pth")`. Inputs to `predict` are raw RGB uint8 NumPy arrays shaped (N,128,128,3). See manifest.json for hashes. This bundle intentionally includes binary weights in Git. Earlier v2 bundles remain preserved.
