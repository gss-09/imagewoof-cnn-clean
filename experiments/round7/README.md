# Round 7: best recipe on a 70/30 re-split

Launched November 8, 2026. New option `--resplit 0.3`: pool the supplied train+val images (10,360) and hold out 30% of each breed, shuffled with seed 42. Result: 7,251 train / 3,109 validation, no overlap; exact lists in `runs/r7_v4_resplit7030/split.json`. Recipe otherwise identical to the 86.72% run (`r5_v4_sgd_wd001`).

| Run | Train / val | EMA best | Flip at best | Final epoch |
| --- | --- | --- | --- | --- |
| `r7_v4_resplit7030` | 7,251 / 3,109 | 83.92% (2,609) @102 | 84.34% | 83.08%, train 99.8% |
| `r5_v4_sgd_wd001` (reference) | 9,065 / 1,295 | 86.72% (1,123) @123 | 86.49% | 85.1% |

Findings: with 20% fewer training images the recipe scored 2.8 points lower and peaked earlier (102 vs 123), consistent with memorization arriving sooner. The validation sets differ, so the gap is a rough estimate of the value of more data, not an exact one. The leaderboard's 70/30 uses the official Imagewoof validation images, which are not available here; this copies the ratio only.
