"""Score each model only on held-out images it never trained on, for each of our validation sets."""
import json, glob, os, re
from pathlib import Path
R = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("EVAL_OUTPUT_DIR", R / "runs" / "alleval"))
NAMES = [l.split()[-1] if len(l.split()) > 1 else l.strip() for l in open(R / "imagewoof_train/classes.txt") if l.strip()]
norm = lambda p: p[p.index("imagewoof_"):]
def split(run):
    d = json.load(open(R / "runs" / run / "split.json")); return {k: {norm(p) for p in v} for k, v in d.items()}
sets = {"supplied val (1,295)": split("r5_v4_sgd_wd001")["validation"], "90/10 held-out (1,036)": split("r8_v4_resplit9010")["validation"],
        "95/5 held-out (518)": split("r10_v4_resplit9505")["validation"], "99/1 held-out (105)": split("r12_v4_resplit9901")["validation"]}
rows = []
for f in sorted(glob.glob(str(OUT / "pred_*.json"))):
    run = Path(f).stem[5:]; pred = json.load(open(f)); train = split(run)["train"]
    label = lambda p: NAMES.index(Path(p).parent.name)
    cells = []
    for s in sets.values():
        clean = [p for p in s if p not in train]
        cells.append(f"{100*sum(pred[p]==label(p) for p in clean)/len(clean):.2f} ({len(clean)})" if clean else "trained on all")
    rows.append((run, cells))
print("model | " + " | ".join(sets)); [print(r + " | " + " | ".join(c)) for r, c in rows]
# head-to-head on images neither r8 nor r12 trained on
both = [p for p in json.load(open(OUT/"pred_r8_v4_resplit9010.json")) if p not in split("r8_v4_resplit9010")["train"] and p not in split("r12_v4_resplit9901")["train"]]
for run in ("r8_v4_resplit9010", "r12_v4_resplit9901"):
    pr = json.load(open(OUT / f"pred_{run}.json")); print("H2H", run, sum(pr[p]==NAMES.index(Path(p).parent.name) for p in both), "/", len(both))
