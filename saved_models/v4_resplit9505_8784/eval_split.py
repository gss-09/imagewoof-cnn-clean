"""Reproduce the 95/5 held-out accuracy: python3 eval_split.py (run from the project root or pass the data root)."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
from solution import CLASS_NAMES, load_model, predict
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
files = json.load(open(here / "split.json"))["validation"]
images = np.stack([np.array(Image.open(root / f).convert("RGB")) for f in files])
labels = np.array([CLASS_NAMES.index(Path(f).parent.name) for f in files])
correct = int((predict(load_model(str(here / "model.pth")), images) == labels).sum())
print(f"{correct}/{len(files)} = {100 * correct / len(files):.2f}%")
