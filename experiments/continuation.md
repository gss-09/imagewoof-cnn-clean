# Experiment 3: small learning-rate restart

Authorized by the user on November 7, 2026. Continue our own DogCNN-v2 from `runs/depth_8conv/model.pth` (epoch 45, 78.1467%, 1,012/1,295). Preserve the source checkpoint.

Use a fresh AdamW optimizer at learning rate 0.0001, cosine decay over at most 20 additional epochs, and early stopping after seven consecutive epochs without a new best. Batch size 32, weight decay 0.0001, seed 42, augmentation, architecture, normalization, and the supplied split remain the same. The source contains no optimizer state; this is a weights continuation with a schedule restart.

The run evaluates the starting weights and saves them as continuation epoch 0. Only strictly higher validation accuracy replaces them. If training never improves, the output retains the starting weights. Continuation epoch numbers count additional epochs, not original training epochs.

```bash
caffeinate -i python3 -u solution.py --architecture DogCNN-v2 --init-checkpoint runs/depth_8conv/model.pth --device mps --output runs/depth_8conv_continue --epochs 20 --batch-size 32 --lr 0.0001 --weight-decay 0.0001 --patience 7 --seed 42 --workers 0 --threads 4
```

CPU smoke check: initialized from trained weights on the 80-image smoke validation subset at 71.25%; one training epoch fell to 70.00%. Reloaded public prediction reproduced the retained 71.25%, and all saved source tensors were verified unchanged. These subset scores are software checks, not full validation results.

After completion, check split equality, full validation through public prediction (automatic), and `python3 verify_pipeline.py runs/depth_8conv_continue/model.pth`. Compare against 78.1467%; hidden-test performance remains unknown.

## Completed result

Early stopping ended the run after seven additional MPS epochs, totaling 5.02 minutes of epoch time. Validation: 76.37%, 75.37%, 77.22%, 76.91%, 75.98%, 76.99%, 76.60%. Training accuracy rose from 84.62% to 86.10%, with loss declining from 0.9356 to 0.9064. No epoch beat the starting 78.1467%.

The saved checkpoint retains continuation epoch 0, and every model tensor exactly matches the source checkpoint. Full public prediction reproduced 78.1467%; split files are byte-identical. This restart provided no validation improvement; the result supports keeping the original epoch-45 model. It does not prove that every possible continuation would fail. No training remains active.

The retained checkpoint passed `verify_pipeline.py`, including isolated fresh-process CPU loading, official labels, preprocessing parity, repeatability, batch behavior, and input validation.

## Cleanup

At the user’s request, the redundant `runs/depth_8conv_continue/model.pth` was removed after tensor equality verification. `checkpoint_reference.json` in that folder points to `runs/depth_8conv/model.pth`; the selected checkpoint is also saved with matching code in `best_model/`. Full continuation logs remain. The smoke folder was removed after its JSON records were archived in `experiments/smoke_checks.json`. For future verification, use `python3 verify_pipeline.py best_model/model.pth`.
