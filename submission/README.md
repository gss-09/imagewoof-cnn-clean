# Group 20 submission

`Group20_Assignment.zip` (SHA-256 dced172a…, 33 MB) contains exactly:

| File | SHA-256 | What |
| --- | --- | --- |
| `solution.py` | 850e9357… | cleaned code: DogCNN class, `load_model`, `predict`, training CLI |
| `model.pth` | 2d8f6073… | 99/1 re-split model, seed 42, EMA weights from epoch 113 (89.52%, 94/105) |

Final checks (Nov 8, 2026) via `experiments/final_check.py`, run on the unzipped archive: all passed on CPU and accelerator. Import trains nothing and needs no PIL/torchvision; `load_model()` default path; eval mode; only allowed layer types; strict weight load; seed 42 / epoch 113 recorded; class order matches both classes.txt; int64 (N,) IDs 0–9; 94/105 reproduced; deterministic; batch-size invariant; single and empty batches; ~11 s (accelerator) / ~63 s (CPU) estimated for 2,591 images.
Retraining with this `solution.py` reproduced the weights bit for bit (`experiments/round15/`).
Still to do on Canvas: upload the zip, comment with every member's name and ID, contribution PDF.
