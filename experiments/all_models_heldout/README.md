# Every saved model scored on held-out images it never trained on (Nov 8, 2026)

Each cell in `results.txt`: accuracy (number of unseen images in that set). Held-out sets are nested (99/1 ⊂ 95/5 ⊂ 90/10). Evaluation only.

r13/r14 (not made by the agent; selected on an external test folder) and all 100/0 models trained on every supplied image, so they cannot be scored.

Head-to-head on the 105 images neither 90/10 nor 99/1 trained on: 90/10 = 90/105 (85.71%), 99/1 = 94/105 (89.52%; this set also chose its epoch, so slightly optimistic).
