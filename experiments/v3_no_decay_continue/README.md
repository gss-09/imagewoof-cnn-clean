# V3 continuation without weight decay

User authorized removing weight decay and continuing with the previously discussed learning rate, interpreted as 0.00001. Initialize from the verified 80.9266% epoch-45 checkpoint in `runs/v3_weight_decay_001/model.pth`. Fresh AdamW optimizer, zero weight decay, cosine schedule over up to 10 extra epochs, patience10. Keep architecture, supplied split, augmentation, batch32 and seed42 unchanged. Output `runs/v3_no_decay_continue`; log `setup/v3_no_decay_continue.log`.

The initialization checkpoint remains preserved; the trainer retains its initial validation score as the score to beat. This experiment changes both weight decay and the training schedule and cannot isolate the effect of removing weight decay alone. It is a continuation, not a fresh run or optimizer-state resume.

## Result

Completed all 10 extra epochs; Python PID2803045 exited. No validation improvement: final extra epoch10 matched 80.9266% (1,048/1,295); previous extra epochs ranged 80.31–80.85%. Best selected epoch remains continuation epoch0 (original epoch45); state tensors are exactly identical to the initialization checkpoint. Final training98.04%. Public prediction reproduced 80.9266%, CPU verifier passed, recorded source and checkpoint hashes match, supplied split unchanged. The original best bundle `saved_models/v3_wd001_8093/` remains selected. No further training launched.
