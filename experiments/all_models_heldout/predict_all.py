"""Predict every supplied train+val image with one model; write {relative_path: predicted_id}."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
code_dir, model_path, root, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
sys.path.insert(0, str(code_dir))
from solution import load_model, predict
files = sorted(str(p.relative_to(root)) for d in ("imagewoof_train/train", "imagewoof_val/val") for p in (root / d).glob("*/*.png"))
model = load_model(model_path)
preds = []
for i in range(0, len(files), 1000):
    imgs = np.stack([np.array(Image.open(root / f).convert("RGB")) for f in files[i:i + 1000]])
    preds.extend(int(x) for x in predict(model, imgs))
json.dump(dict(zip(files, preds)), open(out, "w"))
print(out, len(files))
