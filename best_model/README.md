# Historical DogCNN-v2 bundle

`model.pth` is our DogCNN-v2 checkpoint from epoch 45: **78.1467% validation accuracy (1,012/1,295)**. `solution.py` is the matching self-contained implementation; `requirements.txt` records dependencies. `manifest.json` records file hashes and provenance. `config.json` and `summary.json` are copies of the original training records; their paths refer to the original run.

Load it from this directory:

```python
from solution import load_model, predict
model = load_model("model.pth")
# images: RGB uint8 NumPy array shaped (N, 128, 128, 3)
# predictions = predict(model, images)
```

The checkpoint loads on CPU. See [the process report](../experiments/PROCESS.md) for architecture, data, training, validation, and the unsuccessful continuation. This is a saved model bundle, not a completed course submission: group details and the contribution report are still required.
