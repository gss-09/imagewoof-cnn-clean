# Round 15: retrain the submission model with the cleaned solution.py

The cleaned, submission-only `solution.py` (copy: `solution_clean.py`, SHA-256 850e9357…) keeps only the code path that trained the 99/1 model (`runs/r12_v4_resplit9901`). Defaults are that recipe: 99/1 seeded re-split, 150 epochs, SGD Nesterov lr 0.05 wd 0.001, 5 warm-up + cosine, EMA 0.999, seed 42, workers 6. Launched Nov 8, 2026 as `runs/r15_clean_resplit9901`. A 3-epoch replay matched r12's log to every printed digit and produced an identical split.

## Result (finished Nov 8, 2026)

- Best epoch **113: 89.52% (94/105)**, reloaded `predict()` 89.52%, per-breed accuracy identical to r12.
- Every weight tensor is **bit-identical** to the submitted r12 checkpoint (max abs difference 0.0). The file hash differs (9ea80e54… vs 2d8f6073…) only because the cleaned code saves fewer metadata fields.
- Conclusion: the submitted cleaned `solution.py` reproduces the submitted `model.pth` exactly from seed 42.
- The exact Group20 zip was also tested: loads on CPU, 94/105 on CPU and 94/105 on accelerator after `model.cuda()`.
