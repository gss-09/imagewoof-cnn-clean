# Gentler crop experiment (completed)

Increase RandomResizedCrop minimum area from 0.7 to 0.85; maximum remains 1.0, output 128×128, and default aspect ratio remains unchanged. This may retain more body shape useful for breed classification; The completed CUDA run reached 78.9189% (1,022/1,295), versus the prior MPS result of 78.1467% (1,012/1,295). Hardware also changed, so the gain cannot be attributed to cropping alone.

Train DogCNN-v2 from scratch on the same provided split with seed 42, batch size 32, AdamW learning rate 0.001, weight decay 0.0001, 50 epochs maximum, patience 10, cosine decay, label smoothing 0.1, workers 0, and threads 4. Architecture, horizontal flips, color jitter, normalization, and validation/inference preprocessing remain the same. CUDA replaces MPS, so hardware and augmentation both differ from the previous run; fixed seeds do not ensure identical results across devices.

Run from the project root:

```bash
python -u solution.py --architecture DogCNN-v2 --device cuda --output runs/gentler_crop_085 --epochs 50 --batch-size 32 --lr 0.001 --weight-decay 0.0001 --patience 10 --seed 42 --workers 0 --threads 4
```

Launched November 7, 2026. Log: `setup/gentler_crop_085.log`; output: `runs/gentler_crop_085`. CUDA execution and the saved configuration were confirmed. Completed all 50 epochs; best epoch 47. Reloaded public prediction reproduced 78.9189%. A local copy of the checkpoint and complete run records is now in `runs/gentler_crop_085/`; `solution.py` there preserves the exact training source. Do not use `--init-checkpoint`; this experiment starts from random weights. The new output directory protects existing checkpoints. Compare best validation accuracy and per-class results against 78.1467% (1,012/1,295), and verify the new checkpoint through `verify_pipeline.py` after training. Preserve `best_model/` and historical run configs unchanged until a new result is evaluated.
