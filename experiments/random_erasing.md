# Random erasing experiment (prepared, not launched)

Hypothesis: hiding small patches during training may reduce reliance on individual features and improve generalization. No accuracy improvement is assumed.

Add RandomErasing after normalization: probability 0.25, area fraction 0.02–0.12, aspect ratio 0.5–2.0, fill 0 (mid-gray in normalized coordinates). Training only; original images are not modified. Keep crop 0.85–1.0, horizontal flips, color jitter 0.15, and all inference preprocessing unchanged.

Fresh DogCNN-v2 weights, official split, seed 42, AdamW lr 0.001, weight decay 0.0001, cosine schedule, label smoothing 0.1, batch 32, at most 50 epochs, patience 10, workers 0, threads 4. Compare against the CUDA gentler-crop result of 78.9189% (1,022/1,295). A single seed is exploratory evidence.

Run from the project root:

```bash
python -u solution.py --architecture DogCNN-v2 --device cuda --output runs/random_erasing_025 --epochs 50 --batch-size 32 --lr 0.001 --weight-decay 0.0001 --patience 10 --seed 42 --workers 0 --threads 4
```

Do not initialize from an existing checkpoint. Preserve previous weights. Record best epoch, full validation and per-class accuracy, reproduce public prediction, then run the CPU portability verifier. Before subsequent code edits, save this run's exact `solution.py` alongside its checkpoint; copy the completed run to the laptop and update `experiments/model_registry.json`.

Verification passed: evaluated the actual training/validation transform definitions, checked erasing settings and patch behavior, and confirmed validation excludes erasing. The saved 78.92% checkpoint passed `verify_pipeline.py` on CPU, including isolated-process loading and real-image preprocessing checks. No new training was launched.
