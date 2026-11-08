# Best measured model: 78.92%

DogCNN-v2, gentler crop, best epoch 47; 1,022/1,295 official validation images correct. Exact training source, CPU-loadable checkpoint, original config/summary, dependency specification, and SHA-256 manifest are included. Full validation through public prediction reproduced the recorded score.

From this directory: `from solution import load_model, predict`; then `model = load_model("model.pth")`. Pass RGB uint8 arrays shaped `(N, 128, 128, 3)` to `predict(model, images)`.

This bundle is explicitly tracked in Git, including its binary checkpoint. Other run weights remain ignored. Preserve this bundle when preparing new architectures.
