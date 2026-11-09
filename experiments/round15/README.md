# Round 15: retrain the submission model with the cleaned solution.py

The cleaned, submission-only `solution.py` (copy: `solution_clean.py`, SHA-256 850e9357…) keeps only the code path that trained the 99/1 model (`runs/r12_v4_resplit9901`). Defaults are that recipe: 99/1 seeded re-split, 150 epochs, SGD Nesterov lr 0.05 wd 0.001, 5 warm-up + cosine, EMA 0.999, seed 42, workers 6. Launched Nov 8, 2026 as `runs/r15_clean_resplit9901`. A 3-epoch replay matched r12's log to every printed digit and produced an identical split.
